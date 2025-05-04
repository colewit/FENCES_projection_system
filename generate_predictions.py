import copy
import os
import pickle
import warnings
import concurrent.futures
import traceback

import tqdm # Progress bar utility
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import skewnorm, norm # Statistical distributions and functions
from scipy.interpolate import interp1d # Interpolation functions

from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction techniques
from sklearn.neighbors import NearestNeighbors # Nearest neighbor search
from sklearn.preprocessing import StandardScaler # Data scaling
from sklearn.multioutput import MultiOutputRegressor # Wrapper for multi-output regression
from xgboost import XGBRegressor # Gradient Boosting regressor

# --- Data Preparation Functions ---

def prepare_data(path='../data.csv', use_minor_league_model=True):
    """
    Loads, preprocesses, and merges MLB and MiLB player data.

    Args:
        path (str, optional): Path to the main MLB data CSV file. Defaults to '../data.csv'.
        use_minor_league_model (bool, optional): Flag to determine whether to use a model
            for minor league data conversion or a simpler aggregate method. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing combined and processed player data,
                      with MLB and MiLB stats integrated per player-season.
    """
    print("Preparing data...")

    # Load and basic rename/sort
    data = pd.read_csv(path).rename(columns={'wRC+': 'wRC_plus'})
    data = data.sort_values('Season')

    # Add season number for each player
    data['num_season'] = data.groupby('IDfg').Age.transform(lambda x: range(len(x)))

    # Calculate hitting rates
    data['hit_rate'] = (data['HR'] + data['3B'] + data['2B'] + data['1B']) / data['PA']
    data['HR_rate'] = data.HR / data.PA
    data['1B_rate'] = data['1B'] / data.PA
    data['2B_rate'] = data['2B'] / data.PA
    data['3B_rate'] = data['3B'] / data.PA

    # Calculate next season's rates and stats (shifted)
    data['HR_rate_next'] = data.groupby(['IDfg']).HR_rate.transform(lambda x: x.shift(-1))
    data['1B_rate_next'] = data.groupby(['IDfg'])['1B_rate'].transform(lambda x: x.shift(-1))
    data['2B_rate_next'] = data.groupby(['IDfg'])['2B_rate'].transform(lambda x: x.shift(-1))
    data['3B_rate_next'] = data.groupby(['IDfg'])['3B_rate'].transform(lambda x: x.shift(-1))
    data['hit_rate_next'] = data.groupby(['IDfg'])['hit_rate'].transform(lambda x: x.shift(-1))
    data['OBP_next'] = data.groupby(['IDfg'])['OBP'].transform(lambda x: x.shift(-1))
    data['wRC_plus_next'] = data.groupby('IDfg').wRC_plus.shift(-1)
    data['PA_next'] = data.groupby('IDfg').PA.shift(-1)

    # Add previous season's wRC+
    data['wRC_plus_prev'] = data.groupby('IDfg').wRC_plus.shift(1)

    # Load and merge biographical data
    # Consider adding error handling for file not found
    try:
        bio_data = pd.read_csv('../People.csv', encoding='latin-1')
        bio_data['Name'] = bio_data.nameFirst + ' ' + bio_data.nameLast
        bio_data = bio_data[['Name', 'weight', 'height', 'birthYear']].dropna().sort_values('birthYear')

        player_map = pd.read_csv('player_map.csv') # Maps different player IDs
        data = data.merge(player_map, how='left', on='IDfg')
        data = data.sort_values('birthYear') # Sort for potential merge_asof later

        data = data.merge(bio_data, how='left', on=['Name', 'birthYear'])
    except FileNotFoundError as e:
        warnings.warn(f"Bio data or player map file not found: {e}. Proceeding without bio data.")
        # Add default columns if necessary for downstream code
        if 'weight' not in data.columns: data['weight'] = np.nan
        if 'height' not in data.columns: data['height'] = np.nan
        if 'birthYear' not in data.columns: data['birthYear'] = np.nan # May cause issues if used later

    data['mlb_PA'] = data['PA'] # Store original MLB PA

    # --- Minor League Data Integration ---
    print("Integrating minor league data...")
    try:
        amdf = get_minor_league_data(data, use_model=use_minor_league_model, fit_model=False) # Fit model should likely be False unless retraining here
    except FileNotFoundError as e:
        raise Exception(f"Minor league data file not found: {e}. Proceeding without minor league data.")

    pred_cols = [x for x in amdf.columns if 'pred_' in x] # Columns predicted by the MiLB model

    # Rename predicted columns and add MiLB PA
    amdf = amdf[pred_cols + ['Season', 'Name', 'Age', 'PA', 'AB', 'IDfg']].rename(columns={x: x.replace('pred_', '') for x in pred_cols})
    amdf["milb_PA"] = amdf.PA

    # --- Plus Stat Mapping ---
    # Map MiLB stats to MLB-equivalent "plus" stats using MLB context
    data['rounded_wRC_plus'] = 2 * np.round(data.wRC_plus / 2) # Round for grouping
    amdf['rounded_wRC_plus'] = 2 * np.round(amdf.wRC_plus / 2)

    for col in ['BB%', 'AVG', 'OBP', 'SLG']:
        if col not in data.columns or col not in amdf.columns:
            warnings.warn(f"Column {col} not found in data or amdf. Skipping plus stat mapping for it.")
            continue
        data[f'rounded_{col}'] = data[col].round(2)
        amdf[f'rounded_{col}'] = amdf[col].round(2)
        # Create mapping from rounded stat to plus stat based on MLB season averages
        plus_map = data[['rounded_' + col, col + '+', 'Season']].groupby(['Season', 'rounded_' + col]).agg('mean').round(0).reset_index()
        # Merge the plus stat onto the MiLB data based on season and rounded stat value
        amdf = pd.merge_asof(amdf.sort_values(f'rounded_{col}'), plus_map.sort_values(f'rounded_{col}'), on=f'rounded_{col}', by='Season', direction='nearest')
        pred_cols.append(f'{col}+') # Add the newly mapped plus stat

    # Map MiLB wRC+ to MLB wOBA equivalent
    woba_map = data[['rounded_wRC_plus', 'wOBA', 'Season']].groupby(['Season', 'rounded_wRC_plus']).agg('mean').reset_index()
    amdf = pd.merge_asof(amdf.sort_values('rounded_wRC_plus'), woba_map[['rounded_wRC_plus', 'wOBA', 'Season']]\
                             .sort_values('rounded_wRC_plus'), on='rounded_wRC_plus', by='Season', direction='nearest')
    pred_cols.append('wOBA') # Add mapped wOBA

    # Combine MLB and MiLB data
    data['Level'] = 'MLB'
    amdf['Level'] = 'MiLB'
    data = pd.concat([amdf, data], ignore_index=True) # Use ignore_index

    # --- Handling Players with Both MLB and MiLB Data in Same Season ---
    # Adjust PA for uncertainty: Give less weight to MiLB PA
    # Consider making the divisor a parameter or empirically grounding it
    data['PA'] = np.where(data.Level == 'MLB', data.PA, data.PA / 2)
    data['AB'] = np.where(data.Level == 'MLB', data.AB, data.AB / 2)

    data['multiple_levels'] = data.groupby(['Season', 'IDfg']).Level.transform(lambda x: len(x.unique())) # Check unique levels

    # Define primary level for players appearing at multiple levels in a season
    data['Level'] = np.where(data.mlb_PA > 100, 'MLB', np.where(data.mlb_PA.fillna(0) < data.milb_PA.fillna(0),'MiLB','MLB')) # Prioritize MLB if significant PA, else MiLB if more PA there

    data_to_dedupe = data[data.multiple_levels > 1].copy() # Use .copy()
    data_unique = data[data.multiple_levels <= 1].copy() # Use .copy()

    # Weighted mean function for aggregation
    wm = lambda x: np.average(x.dropna(), weights=data_to_dedupe.loc[x.dropna().index, "PA"]) if not x.dropna().empty else np.nan

    # Define aggregation logic
    agg_columns = list(set([x.replace('pred_', '') for x in pred_cols if x in data_to_dedupe.columns])) # Ensure cols exist
    agg_dict = {col: wm for col in agg_columns if col in data_to_dedupe.columns} # Check existence again
    agg_dict['PA'] = 'sum'
    agg_dict['AB'] = 'sum'

    # Perform weighted aggregation for players at multiple levels
    weighted_avg = data_to_dedupe.groupby(['IDfg', 'Season'], as_index=False).agg(agg_dict)

    # Merge aggregated stats back, dropping original non-aggregated rows for these players
    # Keep only one row per player/season for the deduplicated set
    data_to_dedupe_base = data_to_dedupe.drop(columns=agg_columns + ['PA', 'AB']).drop_duplicates(subset=['IDfg', 'Season'])
    data_to_dedupe = data_to_dedupe_base.merge(weighted_avg, on=['IDfg', 'Season'], how='left')

    # Combine unique players and deduplicated players
    data = pd.concat([data_to_dedupe, data_unique], ignore_index=True)

    # Final cleanup and ID assignment
    data = data.sort_values(['Name', 'Season', 'Level'], ascending=[True, True, False]) # Prioritize MLB level if somehow duplicates remain
    data = data.groupby(['Name', 'Season']).first().reset_index() # Take the first row (prioritizing MLB due to sort)

    # Ensure consistent IDfg for each player across seasons
    data['IDfg'] = data.groupby(['Name'])['IDfg'].transform(lambda x: x.ffill().bfill())

    # Create a composite ID for players missing a standard IDfg
    data['composite_id'] = data['Name'] + data['birthYear'].astype(str)
    data['IDfg'] = data['IDfg'].combine_first(data['composite_id']) # Use composite ID if IDfg is missing

    data['mlb_PA'] = data['mlb_PA'].fillna(0) # Ensure mlb_PA is not NaN

    print("Data preparation finished.")
    return data.sort_values(['IDfg', 'Season']).reset_index(drop=True) # Sort final output


def get_minor_league_data(data, path='minor_league_data.csv', shrinkage_PA=300, use_model=True,
                          target_cols=['wRC_plus', 'hit_rate', 'OBP', 'HR_rate', '2B_rate', 'SLG'],
                          fit_model=False, model_path='minor_league_model.pkl', overwrite=False, minimum_pa=20):
    """
    Loads, processes, and potentially models minor league data to estimate MLB equivalent stats.

    Args:
        data (pd.DataFrame): DataFrame containing MLB data (used for context/merging).
        path (str, optional): Path to the minor league data CSV. Defaults to 'minor_league_data.csv'.
        shrinkage_PA (int, optional): Plate appearance threshold for applying shrinkage (used in non-model approach). Defaults to 300.
        use_model (bool, optional): If True, uses a pre-trained model to predict MLB equivalent stats.
                                     If False, uses an aggregate conversion factor approach. Defaults to True.
        target_cols (list, optional): List of statistical columns to estimate/convert.
                                      Defaults to ['wRC_plus', 'hit_rate', 'OBP', 'HR_rate', '2B_rate', 'SLG'].
        fit_model (bool, optional): If True and use_model is True, retrains the conversion model. Defaults to False.
        model_path (str, optional): Path to save/load the conversion model pickle file. Defaults to 'minor_league_model.pkl'.
        overwrite (bool, optional): If True and fit_model is True, overwrites existing model file. Defaults to False.
        minimum_pa (int, optional): Minimum plate appearances required for a player-season to be considered. Defaults to 20.

    Returns:
        pd.DataFrame: Processed minor league data aggregated per player-season,
                      containing estimated MLB-equivalent stats (prefixed with 'pred_').
    Raises:
        FileNotFoundError: If the minor league data file specified by 'path' is not found.
    """
    print("Processing minor league data...")
    try:
        minor_league_df = pd.read_csv(path).rename(columns={'PlayerId': 'IDfg'})
        minor_league_df = minor_league_df[minor_league_df['PA'] >= minimum_pa].copy() # Filter by min PA early
    except FileNotFoundError:
        print(f"Error: Minor league data file not found at {path}")
        raise # Re-raise the error to be handled by the caller

    if use_model:
        print("Using model for minor league conversions.")
        # Use the modeling function to get predictions

        modeled_data = model_minor_league_conversions(minor_league_df, data, minimum_pa=minimum_pa,
                                                   target_cols=target_cols,
                                                   fit_model=fit_model, model_path=model_path, overwrite=overwrite)
        # Select necessary columns from the modeled output for aggregation
        cols_to_aggregate = [col for col in modeled_data.columns if 'pred_' in col] + ['PA', 'AB', 'Age', 'Name', 'Season', 'IDfg']
        amdf_raw = modeled_data[cols_to_aggregate].copy()

    else:
        print("Using aggregate factors for minor league conversions.")
        # Use aggregate conversion factors based on historical data
        agg_conversion_data = agg_minor_league_conversions(minor_league_df, data, minimum_pa=100) # Use 100 PA for factor calculation
        minor_league_df = minor_league_df.merge(agg_conversion_data, how='left', on=['Level', 'League'])

        # Apply conversion factor
        minor_league_df['wRC_plus_adjusted'] = (minor_league_df['wRC_plus'] * minor_league_df.factor).clip(0, 150) # Apply factor and clip

        # Calculate Age relative to level average
        minor_league_df['Age_for_level'] = minor_league_df.Age - minor_league_df.groupby('Level').Age.transform('median')

        # Heuristic age adjustment: boost younger players for the level
        # TODO: Ground this empirically
        minor_league_df['age_factor'] = (.97) ** (minor_league_df.Age_for_level + 4) # Example adjustment
        minor_league_df['pred_wRC_plus'] = (minor_league_df['wRC_plus_adjusted'] * minor_league_df['age_factor'])

        # Optional: Apply shrinkage towards league average (commented out)
        # amdf['pred_wRC_plus'] = (amdf.factor*100 * (shrinkage_PA-amdf.PA).clip(0,shrinkage_PA) + (amdf.PA).clip(0,shrinkage_PA) * amdf['pred_wRC_plus'])/shrinkage_PA

        # Keep only necessary columns for aggregation
        cols_to_aggregate = ['pred_wRC_plus', 'PA', 'AB', 'Age', 'Name', 'Season', 'IDfg'] # Simplified for non-model path
        amdf_raw = minor_league_df[cols_to_aggregate].copy()


    # --- Aggregate stats for players playing at multiple levels in a season ---
    # Weighted mean function using PA as weight
    wm = lambda x: np.average(x.dropna(), weights=amdf_raw.loc[x.dropna().index, "PA"]) if not x.dropna().empty else np.nan

    pred_cols = [col for col in amdf_raw.columns if 'pred_' in col] # Identify prediction columns
    agg_dict = {col: wm for col in pred_cols} # Apply weighted mean to prediction cols
    agg_dict['PA'] = 'sum'
    agg_dict['AB'] = 'sum'
    agg_dict['Age'] = 'first' # Keep age from the first record (usually highest level due to typical processing order)

    # Group by player and season, aggregate stats
    amdf = amdf_raw.groupby(['Name', 'Season', 'IDfg'], as_index=False).agg(agg_dict)

    print("Minor league data processing finished.")
    return amdf


def agg_minor_league_conversions(minor_league_df, data, minimum_pa=100):
    """
    Calculates simple aggregate conversion factors for MiLB stats to MLB equivalents.

    This function compares players who played in both MiLB and MLB (or AA and AAA)
    in the same season to derive average scaling factors per league/level.

    Args:
        minor_league_df (pd.DataFrame): DataFrame containing minor league stats.
        data (pd.DataFrame): DataFrame containing major league stats.
        minimum_pa (int, optional): Minimum plate appearances in both leagues/levels
                                     for a player-season to be included in factor calculation. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame with columns ['Level', 'League', 'factor'], where 'factor'
                      is the calculated average conversion multiplier for wRC+.
    """
    print("Calculating aggregate minor league conversion factors...")
    mdf = minor_league_df[minor_league_df.PA > minimum_pa].copy() # Filter MiLB data
    data_filtered = data[data.PA > minimum_pa][['wRC_plus', 'Name', 'Season', 'IDfg']].rename(columns={'wRC_plus': 'mlb_wRC_plus'}) # Filter MLB data

    # Merge MLB stats onto MiLB stats for same player/season
    conversion_data = mdf.merge(data_filtered, how='left', on=['Name', 'Season', 'IDfg'])

    # Merge AAA stats onto other MiLB levels (for AA to AAA conversion)
    aaa_stats = mdf[mdf.Level == 'AAA'][['wRC_plus', 'League', 'Name', 'Season', 'IDfg']].rename(columns={'wRC_plus': 'AAA_wRC_plus', 'League': 'AAA_league'})
    conversion_data = conversion_data.merge(aaa_stats, how='left', on=['Name', 'Season', 'IDfg'])

    # Avoid comparing AAA to itself
    conversion_data['AAA_wRC_plus'] = np.where(conversion_data.Level == 'AAA', np.nan, conversion_data['AAA_wRC_plus'])

    # Calculate conversion factors: AAA to MLB, and AA to AAA
    conversion_data['factor'] = np.where(
        conversion_data.Level == 'AA',
        conversion_data['AAA_wRC_plus'] / conversion_data['wRC_plus'], # AA factor is relative to AAA
        conversion_data['mlb_wRC_plus'] / conversion_data['wRC_plus']  # Other levels (AAA) factor relative to MLB
    )

    # Handle potential division by zero or NaNs
    conversion_data['factor'] = conversion_data['factor'].replace([np.inf, -np.inf], np.nan)

    # Aggregate factors by Level and League
    agg_conversion_data = conversion_data.groupby(['Level', 'League'], as_index=False).agg({'factor': 'mean'})

    # Estimate the full AA to MLB factor by chaining AA->AAA and AAA->MLB
    # Calculate the average AAA->MLB factor first
    avg_aaa_factor = agg_conversion_data[agg_conversion_data.Level == 'AAA']['factor'].mean()
    if pd.isna(avg_aaa_factor):
         warnings.warn("Could not calculate average AAA->MLB factor. AA factors might be inaccurate.")
         avg_aaa_factor = 1.0 # Default to 1 if calculation fails

    # Apply the chaining for AA levels
    agg_conversion_data['factor'] = np.where(
        agg_conversion_data.Level == 'AA',
        agg_conversion_data.factor * avg_aaa_factor, # Chain the factors: (AAA/AA) * (MLB/AAA) = MLB/AA
        agg_conversion_data.factor
    ).round(2)

    # Fill any remaining NaNs (e.g., leagues with no conversion data) with a default factor (e.g., 1 or level average)
    agg_conversion_data['factor'] = agg_conversion_data.groupby('Level')['factor'].transform(lambda x: x.fillna(x.mean()))
    agg_conversion_data['factor'] = agg_conversion_data['factor'].fillna(1.0) # Fill any remaining NaNs if a level had no data at all

    print("Aggregate conversion factor calculation finished.")
    return agg_conversion_data


def model_minor_league_conversions(minor_league_df, major_league_df, minimum_pa=400,
                                   target_cols=['wRC_plus', 'hit_rate', 'OBP', 'HR_rate', '2B_rate', 'SLG'],
                                   fit_model=False, model_path='minor_league_model.pkl', overwrite=False):
    """
    Models the conversion of MiLB stats (AA/AAA) to MLB equivalents using XGBoost.

    This function prepares features from AA and AAA performance, merges target MLB
    performance (from the same or next season), trains a multi-output regression model
    (if fit_model is True), and predicts MLB-equivalent stats for all AA/AAA player-seasons.

    Args:
        minor_league_df (pd.DataFrame): DataFrame with minor league stats (must include AA and AAA).
        major_league_df (pd.DataFrame): DataFrame with major league stats.
        minimum_pa (int, optional): Minimum MLB plate appearances required in the target season(s)
                                     to consider the MLB stats reliable. Defaults to 400.
        target_cols (list, optional): List of MLB stats to predict.
        fit_model (bool, optional): Whether to train a new model. Defaults to False.
        model_path (str, optional): Path to load/save the model. Defaults to 'minor_league_model.pkl'.
        overwrite (bool, optional): If True and fit_model is True, overwrites the existing model file. Defaults to False.

    Returns:
        pd.DataFrame: The input minor_league_df (or a relevant subset) with added columns
                      prefixed 'pred_' containing the model's predictions for MLB equivalents.
    """
    print("Modeling minor league conversions...")
    # --- Step 1: Feature Engineering ---
    minor_league_df = minor_league_df.copy()
    major_league_df = major_league_df.copy()

    # Age relative to level average
    minor_league_df['Age_for_level'] = minor_league_df.Age - minor_league_df.groupby('Level').Age.transform('mean')
    # Years spent at the current level
    minor_league_df['years_at_level'] = minor_league_df.groupby(['IDfg', 'Level']).Age.transform(lambda x: range(1, len(x) + 1))

    # Ensure consistent ID types
    minor_league_df['IDfg'] = minor_league_df.IDfg.astype(str)
    major_league_df['IDfg'] = major_league_df.IDfg.astype(str)

    # Define feature columns (ensure these exist in minor_league_df)
    base_feature_cols = ['PA', 'HR_rate', '2B_rate', 'BB%', 'K%', 'BB/K', 'AVG', 'BABIP',
                         'OBP', 'SLG', 'OPS', 'ISO', 'Age_for_level', 'wRC_plus', 'hit_rate',
                         'years_at_level', 'Spd']

    feature_cols = [col for col in base_feature_cols if col in minor_league_df.columns]
    missing_base_cols = set(base_feature_cols) - set(feature_cols)
    if missing_base_cols:
        warnings.warn(f"Missing base feature columns in minor league data: {missing_base_cols}")


    # --- Step 2: Prepare AA and AAA Data ---
    AAA = minor_league_df[minor_league_df.Level == 'AAA'].copy()
    AA = minor_league_df[minor_league_df.Level == 'AA'].copy()

    # Combine AA and AAA stats for the same player/season
    # Use outer merge to keep players who only played at one level
    promotion_data = AAA.merge(AA[['Season', 'IDfg', 'Name', 'Age'] + feature_cols],
                               how='outer', on=['Season', 'IDfg', 'Name', 'Age'],
                               suffixes=['_AAA', '_AA'])
    # Assign 'Level' based on which data was present (prefer AAA if both)
    promotion_data['Level'] = np.where(promotion_data['PA_AAA'].notna(), 'AAA', 'AA')

    # --- Step 3: Prepare Target MLB Data ---
    mlb_stats = major_league_df.sort_values('Season').copy()
    # Calculate rolling PA sum over current and previous season (if available)
    mlb_stats['rolling_PA'] = mlb_stats.groupby('IDfg').PA.transform(lambda x: x.rolling(window=2, min_periods=1).sum())

    # Calculate rolling weighted average for target stats, weighted by PA
    for col in target_cols:
        if col not in mlb_stats.columns:
            warnings.warn(f"Target column '{col}' not found in major_league_df. Skipping.")
            target_cols.remove(col) # Remove from list if not found
            continue
        mlb_stats[f'PA_x_{col}'] = mlb_stats['PA'] * mlb_stats[col]
        mlb_stats[f'rolling_PA_x_{col}'] = mlb_stats.groupby('IDfg')[f'PA_x_{col}'].transform(lambda x: x.rolling(window=2, min_periods=1).sum())
        # Use rolling average only if total PA in window is below threshold
        mlb_stats[col] = np.where(mlb_stats['rolling_PA'] < minimum_pa,
                                  mlb_stats[f'rolling_PA_x_{col}'] / mlb_stats['rolling_PA'].replace(0, np.nan), # Avoid division by zero
                                  mlb_stats[col])

    # Determine max reliable PA achieved by player
    mlb_stats['max_PA'] = mlb_stats.groupby('IDfg').rolling_PA.transform('max')
    # Mark stats as unreliable (NaN) if player never reached minimum_pa threshold in a 2-year window
    for col in target_cols:
        if col in mlb_stats.columns: # Check if column exists
            mlb_stats[col] = np.where(mlb_stats['max_PA'] >= minimum_pa, mlb_stats[col], np.nan)

    mlb_stats = mlb_stats[['IDfg', 'Season', 'max_PA'] + target_cols].copy()
    # Rename target columns for merging
    mlb_stats = mlb_stats.rename(columns={col: f'promotion_{col}' for col in target_cols})
    promotion_cols = [f'promotion_{col}' for col in target_cols] # List of renamed target columns

    # --- Step 4: Merge Target Data (Same Season) ---
    promotion_data = promotion_data.merge(mlb_stats.drop(columns='max_PA'), how='left', on=['IDfg', 'Season'])
    # Merge max_PA separately to keep it for filtering later
    promotion_data = promotion_data.merge(mlb_stats[['IDfg',  'max_PA']].drop_duplicates(), how='left', on=['IDfg']) # Merge max_PA based on season too


    # --- Step 5: Handle Next-Season Promotions ---
    # Identify rows where same-season MLB data was missing or insufficient
    rows_needing_next_season = promotion_data[promotion_data[promotion_cols].isna().any(axis=1)].copy()
    rows_with_this_season = promotion_data[promotion_data[promotion_cols].notna().all(axis=1)].copy()

    # Prepare next season's MLB stats
    next_mlb_stats = mlb_stats.copy()
    next_mlb_stats['Season'] -= 1 # Shift season back to merge with previous MiLB season

    # Merge next season's MLB stats
    rows_needing_next_season = rows_needing_next_season.drop(columns=promotion_cols + ['max_PA'], errors='ignore') # Drop previous merge attempts, ignore errors if columns don't exist
    rows_needing_next_season = rows_needing_next_season.merge(next_mlb_stats, how='left', on=['IDfg', 'Season'])

    # Combine data with same-season targets and next-season targets
    promotion_data = pd.concat([rows_with_this_season, rows_needing_next_season], ignore_index=True)


    # --- Step 6: Impute Targets for Players Never Reaching Majors ---
    promotion_data['max_PA'] = promotion_data['max_PA'].fillna(0)
    # Identify players unlikely to have reached the majors reliably
    promotion_data['never_made_majors'] = (promotion_data['Season'] < 2022) & (promotion_data['max_PA'] < minimum_pa) # Use min_pa threshold

    # Impute with 20th percentile of reliable MLB players for those who likely didn't make it
    reliable_targets = promotion_data[~promotion_data['never_made_majors']][promotion_cols]

    quantile_vals = reliable_targets.quantile(0.2)
    for col in promotion_cols:
        if col in promotion_data.columns: # Check existence
            q_val = quantile_vals.get(col, np.nan) # Use .get for safety
            if not pd.isna(q_val):
                promotion_data.loc[promotion_data['never_made_majors'], col] = promotion_data.loc[promotion_data['never_made_majors'], col].fillna(q_val)


    # Drop rows where target is still NaN after imputation attempts
    # promotion_data = promotion_data.dropna(subset=promotion_cols) # Keep rows even if target is missing for prediction? Or drop?

    # --- Step 7: Final Feature Engineering & Prospect Ranks ---
    # Combine AA/AAA stats with weighted average (example for wRC+)
    promotion_data['PA'] = promotion_data.PA_AA.fillna(0) + promotion_data.PA_AAA.fillna(0)
    # Ensure PA_AAA and PA_AA exist before using them as weights
    pa_aaa = promotion_data.get('PA_AAA', pd.Series(0, index=promotion_data.index)).fillna(0)
    pa_aa = promotion_data.get('PA_AA', pd.Series(0, index=promotion_data.index)).fillna(0)
    total_pa = (pa_aaa + pa_aa).replace(0, np.nan) # Avoid division by zero

    wrc_aaa = promotion_data.get('wRC_plus_AAA', pd.Series(np.nan, index=promotion_data.index)).fillna(0)
    wrc_aa = promotion_data.get('wRC_plus_AA', pd.Series(np.nan, index=promotion_data.index)).fillna(0)

    # Apply league adjustment (e.g., 0.85 for AA relative to AAA) - Make this configurable/empirical
    promotion_data['wRC_plus'] = (wrc_aaa * pa_aaa + 0.85 * wrc_aa * pa_aa) / total_pa

    # Regress towards 100 wRC+ based on PA (sqrt weighting)
    promotion_data['remaining_PA_sqrt'] = np.sqrt((600 - promotion_data.PA).clip(0, 600))
    promotion_data['taken_PA_sqrt'] = np.sqrt(promotion_data.PA.clip(0, 600))
    total_pa_sqrt = (promotion_data.remaining_PA_sqrt + promotion_data.taken_PA_sqrt).replace(0, np.nan)
    promotion_data['wRC_plus_regressed'] = (promotion_data.remaining_PA_sqrt * 100 + promotion_data['wRC_plus'].fillna(100) * promotion_data.taken_PA_sqrt) / total_pa_sqrt

    # Merge Prospect Report Data
    try:
        prospect_report = pd.read_csv('prospect_report_data.csv').drop(columns='Unnamed: 0', errors='ignore')
        promotion_data = promotion_data.merge(prospect_report, how='left', on=['Name', 'Season'])
        # Handle missing prospect ranks, especially for older seasons
        promotion_data['prospect_rank'] = np.where(promotion_data.Season >= 2018, promotion_data.prospect_rank.fillna(-1), promotion_data.prospect_rank) # Fill recent NaNs
        prospect_cols = [x for x in prospect_report.columns if x not in ['Name', 'Season']]
    except FileNotFoundError:
        warnings.warn("Prospect report data not found. Proceeding without prospect features.")
        prospect_cols = []
        # Add placeholder columns if needed downstream
        promotion_data['prospect_rank'] = np.nan

    # Define final feature set for the model
    final_feature_cols = ['Level', 'Age', 'wRC_plus_regressed'] + \
                         [f'{col}_AAA' for col in feature_cols if f'{col}_AAA' in promotion_data.columns] + \
                         [f'{col}_AA' for col in feature_cols if f'{col}_AA' in promotion_data.columns] + \
                         prospect_cols
    # Ensure columns actually exist in the dataframe
    final_feature_cols = [col for col in final_feature_cols if col in promotion_data.columns]


    # --- Step 8: Model Training or Loading ---
    model = None
    X_train_cols = None # To store columns used for training

    if fit_model:
        print("Fitting minor league conversion model...")
        # Prepare training data (drop rows with missing targets)
        train_data = promotion_data.dropna(subset=promotion_cols).copy()
        train_data = train_data[train_data.Season <= 2021] # Example: train on pre-2022 data
        train_data = train_data.fillna(0) # Fill NaNs in features (consider better imputation)

        X_train = pd.get_dummies(train_data[final_feature_cols], dummy_na=False, drop_first=True) # Handle categoricals
        Y_train = train_data[promotion_cols]
        X_train_cols = X_train.columns.tolist() # Save training columns

        # Define the XGBoost model within MultiOutputRegressor
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                verbosity=0, # Quieter output
                n_jobs=-1,   # Use all available CPU cores
                random_state=42,
                # enable_categorical=True # May need specific handling or upstream encoding
            )
        )

        model.fit(X_train, Y_train)
        print("Model training complete.")

        # Save the trained model
        if overwrite or not os.path.exists(model_path):
            print(f"Saving model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f) # Save columns too
        else:
             print(f"Model file {model_path} exists and overwrite is False. Skipping save.")


    
    else: # Load pre-trained model if not fit
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                X_train_cols = model.feature_names_in_
        else:
            raise Exception(f'Cant Find Minor League Model at {model_path}')


    print("Generating predictions...")
    # Prepare data for prediction, ensuring columns match training data
    data_for_pred = promotion_data[final_feature_cols].fillna(0) # Fill NaNs before dummification
    data_for_pred = pd.get_dummies(data_for_pred, dummy_na=False, drop_first=True)
    # Align columns with the training set
    data_for_pred = data_for_pred.reindex(columns=X_train_cols, fill_value=0)

    predictions = model.predict(data_for_pred)

    # Add predictions to the DataFrame
    for i, col in enumerate(target_cols):
        promotion_data[f'pred_{col}'] = predictions[:, i]

    # --- Step 10: Post-Prediction Calculations (Optional) ---
    # Example: Calculate derived stats based on predictions
    promotion_data['pred_HR'] = promotion_data['pred_HR_rate'] * promotion_data.PA
    promotion_data['pred_2B'] = promotion_data['pred_2B_rate'] * promotion_data.PA

    promotion_data['pred_BB'] = promotion_data.PA * (promotion_data['pred_OBP'] - promotion_data['pred_hit_rate'])
    promotion_data['pred_BB%'] = (promotion_data.pred_BB / promotion_data.PA.replace(0,np.nan)).fillna(0) # Avoid div by zero
    promotion_data['AB'] = promotion_data.PA - promotion_data.pred_BB # Calculate AB based on predicted BB
    promotion_data['pred_AVG'] = (promotion_data.pred_hit_rate * promotion_data.PA / promotion_data.AB.replace(0,np.nan)).fillna(0) # Avoid div by zero


    print("Minor league conversion modeling finished.")
    # Return only essential columns + predictions
    return promotion_data


# --- Rolling Calculation Functions ---

def exp_weighted_moment(data, col, weight_col, span):
    """
    Calculates the exponentially weighted mean and standard deviation for a column.

    Weights are determined by the 'weight_col' (e.g., PA) and the decay is
    controlled by 'span'. Handles missing values in the target column by
    excluding them from the calculation.

    Args:
        data (pd.DataFrame): Input DataFrame, must be sorted by player and season.
        col (str): The column for which to calculate the weighted moment.
        weight_col (str): The column to use for weighting (e.g., 'PA', 'AB').
        span (int): The decay factor for the exponential weighting (similar to
                    span in pandas.ewm). Larger span means slower decay.

    Returns:
        pd.DataFrame: The input DataFrame with added columns:
                      f'rolled_{col}' (weighted mean) and
                      f'rolled_std_{col}' (weighted standard deviation).
                      Modifies the DataFrame in-place but also returns it.
    """
    if col not in data.columns:
        warnings.warn(f"Column '{col}' not found for exp_weighted_moment. Skipping.")
        return data
    if weight_col not in data.columns:
         warnings.warn(f"Weight column '{weight_col}' not found for exp_weighted_moment. Skipping '{col}'.")
         return data

    # Create a temporary weight column that's NaN where the stat is NaN
    data[f'PA_for_{col}'] = np.where(data[col].isna(), np.nan, data[weight_col])

    # Calculate EWMA of the weights (denominator)
    # Group by player (IDfg) and apply EWM along the season axis
    rolled_pa_col = data.groupby('IDfg')[f'PA_for_{col}']\
                        .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())

    # Calculate the product of the stat and its weight
    data[f'{col}_x_PA'] = data[col] * data[f'PA_for_{col}'] # Will be NaN if either is NaN

    # Calculate EWMA of the stat*weight product (numerator)
    rolled_stat_pa_col = data.groupby('IDfg')[f'{col}_x_PA']\
                             .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())

    # Calculate the final exponentially weighted mean
    data[f'rolled_{col}'] = rolled_stat_pa_col / rolled_pa_col.replace(0, np.nan) # Avoid division by zero

    # Calculate EWMA Standard Deviation (more complex)
    # Formula involves EWMA of squared values and EWMA of values
    # EW Var(X) = E[X^2] - (E[X])^2
    data[f'{col}_sq_x_PA'] = (data[col]**2) * data[f'PA_for_{col}']
    rolled_stat_sq_pa_col = data.groupby('IDfg')[f'{col}_sq_x_PA']\
                                .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())
    ewma_sq = rolled_stat_sq_pa_col / rolled_pa_col.replace(0, np.nan)
    ewma = data[f'rolled_{col}']
    ewm_var = ewma_sq - (ewma**2)
    # Ensure variance is non-negative due to potential floating point errors
    data[f'rolled_std_{col}'] = np.sqrt(ewm_var.clip(0))

    # Clean up temporary columns
    data = data.drop(columns=[f'PA_for_{col}', f'{col}_x_PA', f'{col}_sq_x_PA'], errors='ignore')

    return data


def get_rolling_columns(data, span=4):
    """
    Applies exponential weighted moment calculations to various stat columns.

    Categorizes stats into 'historic' and 'modern' (Statcast era) and applies
    exp_weighted_moment function appropriately, using 'PA' for historic/general
    stats and 'mlb_PA' for modern stats (as they are typically MLB-only).

    Args:
        data (pd.DataFrame): Input DataFrame with player stats per season.
        span (int, optional): The decay factor (span) for the EWM calculations. Defaults to 4.

    Returns:
        tuple: Contains:
            - pd.DataFrame: The input DataFrame with added 'rolled_' and 'rolled_std_' columns.
            - list: List of original historic and general column names used.
            - list: List of original modern and general column names used.
    """
    print("Calculating rolling statistics...")
    data = data.sort_values(['IDfg', 'Season']).copy() # Ensure correct order for EWM

    # Define column groups (ensure these column names exist in your data)
    general_columns = ['wRC_plus', 'HR_rate', 'ISO+', '2B_rate','OBP','AVG','SLG', 'hit_rate']
    historic_columns = ['BB%+', 'K%+', 'AVG+', 'OBP+', 'SLG+', 'BB/K', 'wOBA']
    modern_columns = ['xBA', 'xSLG', 'xwOBA', 'Barrel%', 'Contact%', 'SwStr%', 'maxEV', 'EV', 'LA'] # Statcast era

    processed_columns_tracker = [] # Track generated columns

    # Process historic and general columns using 'PA' as weight
    cols_to_process_pa = list(set(historic_columns + general_columns))
    print(f"  Calculating rolling stats (using PA) for: {cols_to_process_pa}")
    for col in tqdm.tqdm(cols_to_process_pa, desc="Rolling PA-weighted"):
        data = exp_weighted_moment(data, col, weight_col='PA', span=span)
        processed_columns_tracker.extend([f'rolled_{col}', f'rolled_std_{col}'])


    print(f"  Calculating rolling stats (using mlb_PA) for: {modern_columns}")
    for col in tqdm.tqdm(modern_columns, desc="Rolling mlb_PA-weighted"):
         data = exp_weighted_moment(data, col, weight_col='mlb_PA', span=span)
         processed_columns_tracker.extend([f'rolled_{col}', f'rolled_std_{col}'])

    # Clean up temporary columns potentially left by exp_weighted_moment if errors occurred
    cols_to_drop = [c for c in data.columns if '_x_' in c or 'PA_for_' in c]
    data = data.drop(columns=cols_to_drop, errors='ignore')

    print("Rolling statistics calculation finished.")
    # Return the dataframe and the lists of base columns *actually used*
    return data, list(set(historic_columns + general_columns)), list(set(modern_columns + general_columns))


# --- Player Comparison and Clustering Functions ---
def rolled_wrc_distance(df, baseline, weights):


    baseline = np.array(baseline[-5:])
    n = len(baseline)
    weights = np.array(weights[-n:])
    vals = []
    for i in range(0, len(df)):
        j = max(0, i-(n-1))
        x = df.rolled_wRC_plus.iloc[j:i+1].values
    
        # if there is a mismatch in num seasons (baseline is longer) weigh as 30 pt wRC+ diff
        padding = baseline[:(n-len(x))]-30
        x = np.concatenate([padding, x]) 
        diff = abs(baseline-x)
        weighted_diffs = diff * weights
    
        val = np.mean(weighted_diffs)
        vals.append(val)
        
    df['rolled_wRC_plus_distance'] = vals
    return df

def wrc_distance(df, baseline, weights):

    baseline = np.array(baseline[-5:])
    n = len(baseline)
    weights = np.array(weights[-n:])
    
    vals = []
    for i in range(0, len(df)):
        j = max(0, i-(n-1))
        x = df.wRC_plus.iloc[j:i+1].values
    
        # if there is a mismatch in num seasons (baseline is longer) weigh as 30 pt wRC+ diff
        padding = baseline[:(n-len(x))]-30
        x = np.concatenate([padding, x]) 
        diff = abs(baseline-x)
        weighted_diffs = diff * weights
    
        val = np.mean(weighted_diffs)
        vals.append(val)
        
    df['wRC_plus_distance'] = vals
    return df
        

def find_cluster(data, player, season, max_matches=1000, weights = [0.06, 0.12, 0.16, 0.25, 0.4]):

    '''
    Note that default weights were found using linear regression to find best weights to predict wRC+ based on wRC+ from last 5 years
    '''

    # Get age for player in that season
    pdf = data[(data.Name == player) & (data.Season == season)]

    num_season =  pdf.num_season.iloc[0]
    
    if pdf.empty:
        raise ValueError(f"No data for {player} in season {season}")
    age = pdf.Age.iloc[0]


    age_dict = {18:(0,19), 19:(0,20), 20:(0,22), 21:(0,23), 22:(21,24), 23:(22,25), 24:(23,27), 
                25:(23,28), 26:(24,29), 27:(25,30), 28:(26,31), 29:(27,32), 30:(28, 33),
                31:(29,34),32:(30,35), 33:(31,36), 34:(32,38), 35:(33,39), 36:(34,45),
                37:(35,45),38:(36,45),39:(36,45),40:(37,45),41:(38,45)}

    num_seasons_dict = {0:(0,1), 1:(0,2), 2:(0,3), 3:(1,6), 4:(2,20)}

    lb, ub = age_dict[age]

    # Select comparison players with similar age
    adf = data[data.Age.between(lb, ub)].copy()

    num_season = 0 if pd.isna(num_season) else num_season
    lb, ub = num_seasons_dict[min(num_season, 4)]

    adf = adf[adf.num_season.between(lb, ub)].copy()

    baseline_rolled = data[np.logical_and(data.Name==player, data.Season.between(season-4, season))].rolled_wRC_plus
    baseline = data[np.logical_and(data.Name==player, data.Season.between(season-4, season))].wRC_plus

    
    names = adf.Name.unique()

    rdf = data[data.Name.isin(names)]
    rdf = rdf[np.logical_or(np.logical_and(rdf.Name==player, rdf.Season==season), rdf.Season < season)]

    

    # Suppress only the specific DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        rdf = rdf.groupby('Name').apply(lambda x: rolled_wrc_distance(x, baseline_rolled, weights = weights)).reset_index(drop=True)
        rdf = rdf.groupby('Name').apply(lambda x: wrc_distance(x, baseline, weights = weights)).reset_index(drop=True)
        
    adf = adf.merge(rdf[['Name','rolled_wRC_plus_distance', 'wRC_plus_distance','Season']], how = 'inner', on = ['Name','Season'])

    adf['distance'] = adf.rolled_wRC_plus_distance + adf.wRC_plus_distance
    return adf.sort_values('distance').head(max_matches)



def find_neighbors_for_player(df, historic_columns_base, modern_columns_base, target_name, target_season,
                              target_level, era, n_neighbors=5, n_components=5, feature_importance=None):
    """
    Finds the nearest neighbors for a target player-season using PCA on statistical features.

    Filters data based on 'era' (historic vs. modern Statcast), standardizes features,
    applies PCA, and uses NearestNeighbors to find players with the most similar
    principal component vectors.

    Args:
        df (pd.DataFrame): Input DataFrame (typically the output of find_cluster).
        historic_columns_base (list): Base list of columns relevant for historic comparisons.
        modern_columns_base (list): Base list of columns relevant for modern comparisons.
        target_name (str): Name of the target player.
        target_season (int): Season of the target player.
        target_level (str): Playing level ('MLB' or 'MiLB') of the target player in target_season.
        era (str): 'historic' or 'modern', determines feature set and data filtering.
        n_neighbors (int, optional): Number of neighbors to find. Defaults to 5.
        n_components (int, optional): Number of principal components for PCA. Defaults to 5.
        feature_importance (np.array, optional): Weights to apply to standardized features
                                                   before PCA. Defaults to None (equal weighting).

    Returns:
        tuple: Contains:
            - pd.DataFrame: DataFrame of the nearest neighbors with distance.
            - pd.DataFrame: The filtered DataFrame used for PCA and neighbor search.
            - sklearn.decomposition.PCA: The fitted PCA object.
            - list: The list of feature column names used in the PCA.
            Returns (empty df, None, None, None) if target player not found or insufficient data.
    """
    print(f"  Finding {era}-era neighbors for {target_name} {target_season}...")
    df = df.copy()

    # --- Filter data based on era ---
    if era == 'historic':
        # Exclude seasons with significant Statcast data (post-2014 for MLB)
        # Allow all seasons if target is MiLB (as Statcast isn't reliably available there)
        max_season = 2015 if target_level == 'MLB' else 2030 # Allow future seasons for MiLB sims
        df_filtered = df[np.logical_or(
            (df['Season'] == target_season) & (df['Name'] == target_name),
            df['Season'] < max_season
        )].copy()
        columns_to_use = historic_columns_base
    elif era == 'modern':
        # Require modern data (post-2014)
        if target_season < 2015:
             print(f"    Skipping modern era - target season {target_season} is before 2015.")
             return pd.DataFrame(), df, None, None # Not applicable
        df_filtered = df[np.logical_or(
             (df['Season'] == target_season) & (df['Name'] == target_name),
             df['Season'] >= 2015
        )].copy()
        columns_to_use = modern_columns_base
    else:
        raise ValueError("era must be 'historic' or 'modern'")

    # --- Define Features ---
    # Include base, rolled, and rolled_std versions of the selected columns
    rolled_cols = [f'rolled_{x}' for x in columns_to_use if f'rolled_{x}' in df_filtered.columns]
    std_cols = [f'rolled_std_{x}' for x in columns_to_use if f'rolled_std_{x}' in df_filtered.columns]
    # Add wRC_plus_prev if available
    prev_cols = ['wRC_plus_prev'] if 'wRC_plus_prev' in df_filtered.columns else []

    # Combine and ensure columns exist in the filtered data
    potential_features = list(set(columns_to_use + rolled_cols + std_cols + prev_cols))
    final_feature_cols = [col for col in potential_features if col in df_filtered.columns] # Re-check existence


    # --- Prepare Data for PCA ---
    # Drop rows with NaNs in any feature column
    df_clean = df_filtered.dropna(subset=final_feature_cols).copy()
    df_clean = df_clean.reset_index(drop=True) # Reset index after dropping rows

    # Find the index of the target player in the cleaned data
    target_idx_arr = df_clean[
        (df_clean['Name'] == target_name) & (df_clean['Season'] == target_season)
    ].index

    target_idx = target_idx_arr[0] # Get the first index if found



    # --- Standardization and Feature Weighting ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[final_feature_cols])

    # Apply feature importance weighting if provided
    if feature_importance is not None:
        X = X * feature_importance # Apply weights element-wise
    
    # --- PCA ---
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(X)


    # --- Nearest Neighbors ---
    # Get the PCA vector for the target player
    target_vector = pca_components[target_idx].reshape(1, -1)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1) # Request one extra to include self
    nn.fit(pca_components)

    distances, indices = nn.kneighbors(target_vector)

    # --- Format Output ---
    # Exclude the target player itself from the neighbor list
    neighbor_indices = indices[0][1:]
    neighbor_distances = distances[0][1:]

    neighbors_df = df_clean.iloc[neighbor_indices].copy()
    neighbors_df['DistanceToTarget'] = neighbor_distances

    # Select relevant columns to return
    cols_to_keep_base = ['Name', 'Season', 'IDfg', 'Age', 'Level', 'DistanceToTarget']
    # Add target variable if present
    if 'wRC_plus_next' in neighbors_df.columns: cols_to_keep_base.append('wRC_plus_next')
    # Add features used
    cols_to_keep = list(set(cols_to_keep_base + final_feature_cols))
    # Add rate columns needed for projection simulation
    rate_cols = ['1B_rate', '2B_rate', '3B_rate', 'HR_rate', 'hit_rate', 'OBP', 'PA', 'PA_next']
    cols_to_keep.extend([rc for rc in rate_cols if rc in neighbors_df.columns and rc not in cols_to_keep])

    # Ensure all selected columns exist before returning
    final_cols_to_keep = [col for col in cols_to_keep if col in neighbors_df.columns]

    print(f"    Found {len(neighbors_df)} {era}-era neighbors.")
    return neighbors_df[final_cols_to_keep], df_filtered, pca, final_feature_cols


# --- Distribution Building and Projection Functions ---

def build_distribution_for_player(data, player, season, historic_columns, modern_columns,
                                  beta=0.3, alpha=0.5, show_plots=True,
                                  modern_feature_importances=None, historic_feature_importances=None):
    """
    Builds a projected distribution of next-season wRC+ for a player.

    Finds similar players (cluster + nearest neighbors), analyzes their next-season
    wRC+ relative to their rolled wRC+, fits a skewed normal distribution to these
    deltas, adjusts the mean based on a generic aging curve, and returns samples
    from the final projected distribution.

    Args:
        data (pd.DataFrame): Full dataset with calculated rolling stats.
        player (str): Target player name.
        season (int): Target season.
        historic_columns (list): List of base historic features.
        modern_columns (list): List of base modern features.
        beta (float, optional): Weighting factor for combining distributions from
                                the broader cluster vs. closer neighbors (0 to 1).
                                Higher beta gives more weight to the broader cluster. Defaults to 0.3.
        alpha (float, optional): Weighting factor for blending the model-derived mean delta
                                 with a generic aging curve delta (0 to 1). Higher alpha
                                 gives more weight to the model/comps. Defaults to 0.5.
        show_plots (bool, optional): Whether to display histograms and fitted distributions. Defaults to True.
        modern_feature_importances (np.array, optional): Feature weights for modern neighbor search.
        historic_feature_importances (np.array, optional): Feature weights for historic neighbor search.


    Returns:
        tuple: Contains:
            - np.array: Samples from the projected next-season wRC+ distribution.
            - pd.DataFrame: DataFrame of the initial cluster of similar players (used for neighbor search).
            - pd.DataFrame: DataFrame of the nearest neighbors found (combined historic/modern).
            - tuple: Parameters (a, loc, scale) of the final fitted skew-normal distribution delta.
                     Returns (None, None, None, (np.nan, np.nan, np.nan)) if process fails.
    """
    print(f"Building Distribution for {player} {season}...")
    data = data.sort_values(['IDfg', 'Season']).copy()

    # Ensure target variable exists (or create if needed, though it should be pre-calculated)
    if 'wRC_plus_next' not in data.columns:
        data['wRC_plus_next'] = data.groupby('IDfg')['wRC_plus'].shift(-1)

    # --- Find Similar Players ---
    cdf = find_cluster(data, player, season) # Find broad cluster based on wRC+ trajectory


    player_row_df = data[(data.Name == player) & (data.Season == season)]
    if player_row_df.empty:
         print(f"  Target player {player} {season} not found in main data. Cannot proceed.")
         return None, cdf, None, (np.nan, np.nan, np.nan)
    player_row = player_row_df.iloc[0]

    level = player_row.Level # Get player level (MLB/MiLB) for neighbor search logic
    target_rolled_wrc = player_row.rolled_wRC_plus


    # --- Find Nearest Neighbors (Refined Similarity) ---
    # Define number of neighbors based on level (more for MiLB due to higher variance/less data)
    historic_neighbors_n = 50 if level == 'MLB' else 100
    modern_neighbors_n = 50 if level == 'MLB' else 0 # Only use modern for MLB players

    # Initialize DataFrames
    ndf_historic = pd.DataFrame()
    ndf_modern = pd.DataFrame()
    cdf_h = cdf # Base cluster data used for historic search
    cdf_m = cdf # Base cluster data used for modern search (will be filtered by find_neighbors)

    print(historic_columns, modern_columns)
    ndf_historic, cdf_h, _, _ = find_neighbors_for_player(cdf, historic_columns, modern_columns,
                                                          target_name=player, target_season=season,
                                                          target_level=level, era='historic',
                                                          n_neighbors=historic_neighbors_n, n_components=5,
                                                          feature_importance=historic_feature_importances)

    # Find modern neighbors only if the player is MLB and season allows
    if level == 'MLB' and season >= 2015 :
        ndf_modern, cdf_m, _, _ = find_neighbors_for_player(cdf, historic_columns, modern_columns,
                                                            target_name=player, target_season=season,
                                                            target_level=level, era='modern',
                                                            n_neighbors=modern_neighbors_n, n_components=5,
                                                            feature_importance=modern_feature_importances)

    # Combine neighbor sets and cluster data used
    ndf = pd.concat([ndf_modern, ndf_historic], ignore_index=True)
    # Use the potentially filtered cluster data from neighbor search steps
    # Combine the dataframes that were *actually used* as input for the neighbor searches
    cdf_combined = pd.concat([cdf_m, cdf_h], ignore_index=True).drop_duplicates(subset=['Name', 'Season', 'IDfg']) # Use IDfg too


    # --- Analyze Next Season Performance of Comps ---
    # Calculate the change in wRC+ from rolled_wRC+ for comps
    cdf_combined = cdf_combined.dropna(subset=['wRC_plus_next', 'rolled_wRC_plus'])
    ndf = ndf.dropna(subset=['wRC_plus_next', 'rolled_wRC_plus'])


    # Deltas for the broader cluster
    x_deltas = (cdf_combined['wRC_plus_next'] - cdf_combined['rolled_wRC_plus']).astype(float)
    # Deltas for the nearest neighbors
    y_deltas = (ndf['wRC_plus_next'] - ndf['rolled_wRC_plus']).astype(float)


    # --- Fit Skew-Normal Distributions ---
    # Fit to broader cluster 
    a_fit1, loc_fit1, scale_fit1 = skewnorm.fit(x_deltas if not x_deltas.empty else y_deltas)
    
    # Fit to nearest neighbors 
    a_fit2, loc_fit2, scale_fit2 = skewnorm.fit(y_deltas if not y_deltas.empty else x_deltas)


    # --- Combine Distributions and Apply Aging Curve ---
    # Blend samples from the two distributions based on beta
    # Beta weights the broader cluster (cdf), (1-beta) weights neighbors (ndf)
    n_total_samples = 10000 # Number of samples for intermediate fitting
    n1 = int(n_total_samples * beta)
    n2 = int(n_total_samples * (1 - beta))

    samples1 = skewnorm.rvs(a_fit1, loc=loc_fit1, scale=scale_fit1, size=n1)
    samples2 = skewnorm.rvs(a_fit2, loc=loc_fit2, scale=scale_fit2, size=n2)
    combined_samples = np.concatenate([samples1, samples2])

    # Fit a final skew-normal distribution to the combined samples
    a_fit_comb, loc_fit_comb, scale_fit_comb = skewnorm.fit(combined_samples)


    # Load generic aging curve data
    try:
        aging_curve = pd.read_csv('aging_curve.csv')
        aging_curve_shift = aging_curve.loc[aging_curve.Age == player_row.Age, 'delta'].iloc[0]
    except (FileNotFoundError, IndexError, KeyError, AttributeError): # Added AttributeError
        warnings.warn("Aging curve data not found or player age missing/invalid. Using 0 for aging adjustment.")
        aging_curve_shift = 0

    # Calculate mean of the combined delta distribution
    delta_param = a_fit_comb / np.sqrt(1 + a_fit_comb**2)
    mu_comb = loc_fit_comb + scale_fit_comb * delta_param * np.sqrt(2 / np.pi)

    # Blend the model-derived mean delta with the aging curve delta using alpha
    # Alpha weights the model/comps, (1-alpha) weights the generic aging curve
    mu_adjusted = alpha * mu_comb + (1 - alpha) * aging_curve_shift

    # Adjust the location parameter of the combined distribution to match the new blended mean
    loc_final_adjusted = mu_adjusted - scale_fit_comb * delta_param * np.sqrt(2 / np.pi)

    # --- Generate Final Projection Samples ---
    final_params = (a_fit_comb, loc_final_adjusted, scale_fit_comb)
    # Generate samples representing the change from the player's rolled_wRC+
    delta_samples = skewnorm.rvs(*final_params, size=1000) # Generate 1000 samples for final output
    # Add the player's rolled wRC+ to get the projected wRC+ distribution
    projected_wrc_samples = delta_samples + target_rolled_wrc

    # --- Plotting (Optional) ---
    if show_plots:
        print("  Plotting distributions...")
        plt.figure(figsize=(12, 5))

        # Plot 1: Delta distributions
        plt.subplot(1, 2, 1)
        # Determine plot limits dynamically
        min_delta = min(x_deltas.min() if not x_deltas.empty else -100, y_deltas.min() if not y_deltas.empty else -100) - 10
        max_delta = max(x_deltas.max() if not x_deltas.empty else 100, y_deltas.max() if not y_deltas.empty else 100) + 10
        ln_space = np.linspace(min_delta, max_delta, 500)

        # Only plot if data exists
        if not x_deltas.empty:
             plt.hist(x_deltas, bins=30, density=True, alpha=0.3, label=f"Cluster Deltas (n={len(x_deltas)})")
        if not y_deltas.empty:
             plt.hist(y_deltas, bins=30, density=True, alpha=0.3, label=f"Neighbor Deltas (n={len(y_deltas)})")

        pdf_final_delta = skewnorm.pdf(ln_space, *final_params)
        plt.plot(ln_space, pdf_final_delta, label="Final Adjusted Delta PDF", color='red', linewidth=2)
        plt.title(f"{player} {season} - Delta wRC+ Distribution (NextYr - Rolled)")
        plt.xlabel("wRC+ Change")
        plt.ylabel("Density")
        plt.xlim(max(min_delta, -150), min(max_delta, 150)) # Limit x-axis for readability
        plt.legend()
        plt.grid(True, alpha=0.3)


        # Plot 2: Final Projected wRC+ Distribution
        plt.subplot(1, 2, 2)
        plt.hist(projected_wrc_samples, bins=30, density=True, alpha=0.6, label="Projected wRC+")
        plt.axvline(target_rolled_wrc, color='blue', linestyle='--', label=f"Current Rolled wRC+ ({target_rolled_wrc:.0f})")
        if 'wRC_plus_next' in player_row and not pd.isna(player_row.wRC_plus_next):
            plt.axvline(player_row.wRC_plus_next, color='red', linestyle='-', label=f"Actual Next wRC+ ({player_row.wRC_plus_next:.0f})")
        plt.title(f"{player} {season+1} - Projected wRC+")
        plt.xlabel("Projected wRC+")
        plt.ylabel("Density")
        # Determine xlim dynamically for projection plot
        min_proj = projected_wrc_samples.min()
        max_proj = projected_wrc_samples.max()
        plt.xlim(min_proj - 10, max_proj + 10)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("Distribution building finished.")
    # Return the original cluster df (cdf) and combined neighbors (ndf)
    return projected_wrc_samples, cdf, ndf, final_params


def get_baseline_projections(data, similar_players_df, player, season, dist_params, n_samples=1000):
    """
    Generates baseline projections for rate and counting stats based on similar players,
    using Cholesky decomposition for correlations and skew-normal/empirical distributions.

    Simulates correlated changes for key rate statistics relative to the target player's
    rolled baseline stats, using historical data from similar players. Projects Plate
    Appearances (PA) based on its correlation with wRC+ change. Calculates projected
    counting stats and traditional slash line stats (AVG/OBP/SLG/OPS).

    Args:
        data (pd.DataFrame): Full dataset with historical and rolled stats.
        similar_players_df (pd.DataFrame): DataFrame with 'Name' and 'Season' of similar players.
        player (str): Name of the target player.
        season (int): The season *from which* the projection is based (projects for season + 1).
        dist_params (tuple): Parameters (a, loc, scale) of the fitted skew-normal distribution
                             for wRC+ change.
        n_samples (int, optional): Number of simulation samples to generate. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with projection samples for the target player's next season.
                      Columns include projected rates, counting stats, PA, AVG, OBP, SLG, OPS, etc.
    """
    print(f"Generating Projections for {player} {season}")

    # --- Initial Calculation of Change Columns ---
    # Note: Calculates change = Value(T+1) - RolledValue(T) using potentially pre-calculated '_next' columns.
    data['doubles_rate_change'] = data['2B_rate_next'] - data.rolled_2B_rate
    data['OBP_change'] = data['OBP_next'] - data.rolled_OBP
    data['hit_rate_change'] = data['hit_rate_next'] - data.rolled_hit_rate
    data['HR_rate_change'] = data['HR_rate_next'] - data.rolled_HR_rate
    data['wRC_plus_change'] = data['wRC_plus_next'] - data.rolled_wRC_plus

    # === Helper function to create an inverse CDF (Quantile Function) ===
    def make_inv_cdf(real_deltas):
        """Creates an interpolation function representing the inverse CDF."""
        sorted_vals = np.sort(real_deltas)
        ecdf = np.linspace(0, 1, len(sorted_vals))
        # Create interpolation function: probability -> value
        return interp1d(ecdf, sorted_vals, bounds_error=False,
                        fill_value=(sorted_vals[0], sorted_vals[-1])) # Extrapolate using min/max

    # --- Recalculation of _next and _change columns ---
    # Comment in original code: "make these sttart from diff from rolled not diff from last yr"
    data = data.sort_values('Season') # Ensure data is sorted for .shift()

    # Recalculate _next columns using shift(-1) to get Value(T+1) based on Value(T)
    data['HR_rate_next'] = data.groupby(['Name']).HR_rate.transform(lambda x: x.shift(-1))
    data['2B_rate_next'] = data.groupby(['Name'])['2B_rate'].transform(lambda x: x.shift(-1))
    data['OBP_next'] = data.groupby(['Name'])['OBP'].transform(lambda x: x.shift(-1))
    data['hit_rate_next'] = data.groupby(['Name'])['hit_rate'].transform(lambda x: x.shift(-1))
    # Note: 1B_rate_next and 3B_rate_next are not recalculated

    # Recalculate _change columns using the newly shifted _next values
    data['doubles_rate_change'] = data['2B_rate_next'] - data.rolled_2B_rate
    data['HR_rate_change'] = data['HR_rate_next'] - data.rolled_HR_rate
    data['wRC_plus_change'] = data.wRC_plus_next - data.rolled_wRC_plus
    # Comment in original code: "change to rolled"
    data['OBP_change'] = data['OBP_next'] - data.rolled_OBP
    data['hit_rate_change'] = data['hit_rate_next'] - data.rolled_hit_rate

    # --- Prepare Data from Similar Players ---
    # Merge the relevant change stats for similar players
    rate_df = similar_players_df[['Name', 'Season', 'PA', 'wRC_plus']].merge(
        data[['Name', 'Season', 'HR_rate_change', 'wRC_plus_change', 'doubles_rate_change', 'OBP_change', 'hit_rate_change']],
        how='left', on=['Name', 'Season']
    )
    # Drop rows where any of these key change stats are missing
    rate_df = rate_df.dropna()

    # === Simulation Inputs ===
    a_fit, loc_fit, scale_fit = dist_params # Unpack skew-normal parameters for ΔwRC+

    # Comment in original code: "Empirical diffs"

    # Construct correlation matrix for key rate changes
    # Note: Uses hit_rate_change, not a separate AVG change
    corr_matrix = rate_df[['wRC_plus_change', 'hit_rate_change', 'OBP_change',
                           'HR_rate_change', 'doubles_rate_change']].corr().values

    # Calculate correlation between PA and wRC+ change
    corr_PA = rate_df[['wRC_plus_change', 'PA']].corr().iloc[0, 1]

    # Merge again? Seems potentially redundant, but preserving original code.
    rate_df = rate_df.merge(similar_players_df[['Name', 'Season']], how='inner')

    # Extract the actual delta values from similar players for inverse CDFs
    real_delta_wrc = rate_df['wRC_plus_change']
    real_delta_avg = rate_df['hit_rate_change'] # Corresponds to hit_rate_change in corr matrix
    real_delta_obp = rate_df['OBP_change']
    real_delta_hr = rate_df['HR_rate_change']
    real_delta_2b = rate_df['doubles_rate_change']
    real_pa = rate_df['PA'] # PA values from similar players in season T

    # === Cholesky Decomposition and Correlated Sampling ===
    # Decompose the correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    # Generate independent standard normal random samples
    z = np.random.randn(n_samples, 5) # 5 variables in corr_matrix

    # --- Project PA ---
    # Comment block in original code: ''' Start PA finder'''

    # Project PA based on its correlation with the first variable (wRC_plus_change)
    # Comment in original code: "note this is done on corr to change in wRC+ but should be changed to wRC+ after delta"
    z_pa = corr_PA * z[:, 0] + np.sqrt(1 - corr_PA**2) * np.random.randn(n_samples)

    # Create inverse CDF for historical PA values
    sorted_pa = np.sort(real_pa)
    ranks = np.linspace(0, 1, len(sorted_pa))
    inverse_cdf_pa = interp1d(ranks, sorted_pa, bounds_error=False, fill_value=(sorted_pa[0], sorted_pa[-1]))

    # Transform normal PA component to uniform, then sample from inverse PA CDF
    u2 = norm.cdf(z_pa) # Note: Uses u2 variable name, but relates to z_pa
    PAs = inverse_cdf_pa(u2) # These are the projected PAs

    # Comment block in original code: ''' End PA finder'''

    # Generate correlated standard normal samples
    z_corr = z @ L.T # Apply Cholesky factor

    # === Transform Correlated Samples back to Original Delta Distributions ===

    # Step 2: Transform correlated normal samples (1st dim) to ΔwRC+ using skew-normal PPF
    u1 = norm.cdf(z_corr[:, 0])
    delta_wrc = skewnorm.ppf(u1, a=a_fit, loc=loc_fit, scale=scale_fit)

    # Step 3: Transform correlated normal samples (2nd, 3rd dim) to Δhit_rate and ΔOBP
    # Comment in original code: "Transform ΔAVG and ΔBB%" - Uses hit_rate and OBP deltas
    inverse_cdf_avg = make_inv_cdf(real_delta_avg) # Inverse CDF for hit_rate_change
    inverse_cdf_obp = make_inv_cdf(real_delta_obp) # Inverse CDF for OBP_change

    u2 = norm.cdf(z_corr[:, 1]) # Uses u2 again
    u3 = norm.cdf(z_corr[:, 2])
    delta_avg = inverse_cdf_avg(u2) # This is projected hit_rate_change
    delta_obp = inverse_cdf_obp(u3) # This is projected OBP_change

    # Step 4: Transform correlated normal samples (4th dim) to ΔHR_rate
    # Comment in original code: "HR% based on wRC+ and AVG" - Uses HR_rate delta
    z_hr = z_corr[:, 3]
    u4 = norm.cdf(z_hr)
    inverse_cdf_hr = make_inv_cdf(real_delta_hr)
    delta_hr = inverse_cdf_hr(u4) # This is projected HR_rate_change

    # Step 5: Transform correlated normal samples (5th dim) to Δ2B_rate
    # Comment in original code: "2B% based on wRC+, AVG, HR%" - Uses 2B_rate delta
    z_2b = z_corr[:, 4]
    u5 = norm.cdf(z_2b)
    inverse_cdf_2b = make_inv_cdf(real_delta_2b)
    delta_2b = inverse_cdf_2b(u5) # This is projected 2B_rate_change

    # --- Calculate Final Projected Stats ---
    # Get the target player's baseline rolled stats from Season T
    pdf = data[(data.Name == player) & (data.Season == season)]

    # Create a dictionary to store projection results for each sample
    projection_dict = {}
    # Get baseline rolled values from the player's data for the target season
    projection_dict['rolled_wRC_plus'] = float(pdf['rolled_wRC_plus'].iloc[0]) # Store baseline
    projection_dict['wRC_plus'] = float(pdf['rolled_wRC_plus'].iloc[0]) + delta_wrc # Add simulated change
    projection_dict['hit_rate'] = float(pdf['rolled_hit_rate'].iloc[0]) + delta_avg # Add simulated change
    projection_dict['HR_rate'] = float(pdf['rolled_HR_rate'].iloc[0]) + delta_hr # Add simulated change
    projection_dict['2B_rate'] = float(pdf['rolled_2B_rate'].iloc[0]) + delta_2b # Add simulated change
    # Assume 3B rate change is 0 (wasn't in correlation matrix) - use raw rate from season T as baseline
    projection_dict['3B_rate'] = float(pdf['3B'].fillna(0).iloc[0] / pdf.PA.iloc[0]) + 0
    # Calculate projected 1B rate by subtraction
    projection_dict['1B_rate'] = projection_dict['hit_rate'] - projection_dict['HR_rate'] - projection_dict['3B_rate'] - projection_dict['2B_rate']

    # Project OBP by adding simulated change to baseline
    projection_dict['OBP'] = float(pdf['rolled_OBP'].iloc[0]) + delta_obp

    # Calculate projected BB rate (includes HBP implicitly if OBP does)
    projection_dict['BB_rate'] = projection_dict['OBP'] - projection_dict['hit_rate']

    # Calculate projected counting stats using projected rates and projected PA ('PAs')
    projection_dict['BB'] = projection_dict['BB_rate'] * PAs
    projection_dict['HR'] = projection_dict['HR_rate'] * PAs
    projection_dict['2B'] = projection_dict['2B_rate'] * PAs
    projection_dict['3B'] = projection_dict['3B_rate'] * PAs
    projection_dict['1B'] = projection_dict['1B_rate'] * PAs
    projection_dict['H'] = projection_dict['hit_rate'] * PAs

    # Calculate projected PA and AB
    projection_dict['PA'] = PAs
    projection_dict['AB'] = PAs - projection_dict['BB'] # Approximation: AB = PA - BB

    # Calculate projected AVG (H/AB) - potential division by zero if AB is 0
    projection_dict['AVG'] = projection_dict['H'] / projection_dict['AB']

    # Calculate projected Total Bases (TB)
    projection_dict['TB'] = (4 * projection_dict['HR'] + 3 * projection_dict['3B'] +
                             2 * projection_dict['2B'] + 1 * projection_dict['1B'])
    # Calculate projected SLG (TB/AB) - potential division by zero
    projection_dict['SLG'] = projection_dict['TB'] / projection_dict['AB']

    # Calculate projected OPS (OBP + SLG)
    projection_dict['OPS'] = projection_dict['OBP'] + projection_dict['SLG']

    # Create the final DataFrame from the dictionary of arrays
    projection_df = pd.DataFrame(projection_dict)

    # Add identifiers: player name, projection season, sample number
    projection_df['Name'] = player
    projection_df['Season'] = season + 1 # Projection is for the *next* season
    projection_df['Sample'] = range(1, 1 + len(projection_df)) # Add sample number (1 to n_samples)

    # Select and order columns, round numeric columns to 3 decimal places
    projection_df = projection_df[['Name', 'Season', 'Sample'] + [x for x in projection_df if x not in ['Name', 'Season', 'Sample']]].round(3)

    return projection_df

    # --- Helper Functions for Parallel Processing ---
def process_player(args):
    """
    Worker function to build distribution and get projections for a single player-season.

    Args:
        args (tuple): A tuple containing (player_name, season).

    Returns:
        pd.DataFrame or None: DataFrame with projections for the player, or None if an error occurs.
    """
    try:
        player, season, data, historic_columns, modern_columns = args

        # Build the wRC+ distribution
        # Note: show_plots=False is crucial for parallel execution
        samples, cluster_df, neighbors_df, dist_params = build_distribution_for_player(
            data, player=player, season=season,
            historic_columns=historic_columns, modern_columns=modern_columns,
            show_plots=False # Disable plotting in parallel workers
        )

        projection_df = get_baseline_projections(data, cluster_df, player, season, dist_params, n_samples=1000)
        return projection_df
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return pd.DataFrame()

def save_chunk(predictions_list, mode='a', header=False):
    """
    Concatenates and saves a chunk of projection results to the output CSV.

    Args:
        predictions_list (list): List of DataFrames (or None) from process_player.
        mode (str, optional): File writing mode ('a' for append, 'w' for write). Defaults to 'a'.
        header (bool, optional): Whether to write the header row. Defaults to False.
    """
    # Filter out None results (from errors) before concatenating
    valid_predictions = [df for df in predictions_list if df is not None and not df.empty]
    if valid_predictions:
        try:
            df = pd.concat(valid_predictions, ignore_index=True)
            df.to_csv(output_file, mode=mode, index=False, header=header)
        except Exception as e:
             print(f"Error saving chunk to {output_file}: {e}")
    else:
        print("  No valid predictions in this chunk to save.")
        
# --- Main Execution Block ---

if __name__ == '__main__':

    import traceback
    # --- Configuration ---
    LOAD_FROM_CACHE = True # Set to True to load pre-processed data and columns
    OVERWRITE = False    # Set to True to overwrite cache files if LOAD_FROM_CACHE is False
    chunk_size = 10         # Number of players to process in each parallel batch
    output_file = 'predictions.csv' # File to save the final projection results
    data_file_path = '../data.csv'  # Path to the raw input data

    # --- Define Players to Project ---
    # Example: Select players active in 2024 with more than 50 PA
    # Adjust the filtering criteria as needed
    target_projection_year = 2023 # Projecting for players active in this year
    min_pa_threshold = 50
    
    # --- Data Loading and Preprocessing ---
    if LOAD_FROM_CACHE:
        print("Loading data and columns from cache...")
        
        # Load pre-processed data
        data = pd.read_csv('cached_data_obj.csv')
        # Load pre-calculated column lists
        with open('columns.pkl', 'rb') as f:
            column_data = pickle.load(f)

            historic_columns = column_data['historic'] 
            modern_columns = column_data['modern']
    else:
        print("Processing data from scratch...")
        # Prepare the data (loads raw data, integrates MiLB, etc.)
        # Ensure paths inside prepare_data are correct relative to execution location
        data = prepare_data(path=data_file_path) # Pass path explicitly

        # Calculate rolling statistics and get column lists
        data, historic_columns, modern_columns = get_rolling_columns(data)

        # Save processed data and column lists to cache if overwrite is enabled
        if OVERWRITE:
            print("Saving processed data and columns to cache...")

            # Save column lists
            column_data_to_save = {'historic': historic_columns, 'modern': modern_columns}
            with open('columns.pkl', 'wb') as f:
                pickle.dump(column_data_to_save, f)
            # Save processed data
            data.to_csv('cached_data_obj.csv', index=False)
            print("Cached data saved.")



    print(f"Selecting players from {target_projection_year} with > {min_pa_threshold} PA...")
    smpl = data[(data.PA > min_pa_threshold) & (data.Season == target_projection_year)].copy()

    # Create list of (player_name, season) tuples to process
    # Ensure no duplicates
    player_seasons = list(smpl[['Name', 'Season']].drop_duplicates().itertuples(index=False, name=None))
    print(f"Found {len(player_seasons)} player-seasons to process.")





    # --- Parallel Execution ---
    # Remove old output file if it exists to prevent appending to previous runs
    if os.path.exists(output_file):
        print(f"Removing existing output file: {output_file}")
        try:
            os.remove(output_file)
        except OSError as e:
            print(f"Error removing file {output_file}: {e}. Appending might occur.")

    print(f"Starting parallel processing in chunks of {chunk_size}...")
    # Process players in chunks and save periodically
    all_results = [] # Optional: collect all results if needed later, might use lots of memory

    for i in tqdm.tqdm(range(0, len(player_seasons), chunk_size), desc="Processing Chunks"):
        # Get the current chunk of player-season tuples
        #chunk = player_seasons[i:i + chunk_size]
        chunk = [(player, season, data, historic_columns, modern_columns) for (player, season) in player_seasons[i:i+chunk_size]]


        # Use ProcessPoolExecutor for parallel execution
        # 'with' statement ensures resources are properly managed
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # map applies process_player to each item in chunk concurrently
            # list() collects the results (order is preserved)
            results_chunk = list(executor.map(process_player, chunk))

            if isinstance(results_chunk[0], dict):
                print(results_chunk[0]['traceback'])
                raise Exception('ok')

        # Save the results of the current chunk
        print(f"\nSaving results for chunk {i//chunk_size + 1}...")
        # Write header only for the very first chunk (i == 0)
        save_chunk(results_chunk, mode='a', header=(i == 0))
        all_results.extend(results_chunk) # Optional: append to master list

    print(f"\nParallel processing complete. Results saved to {output_file}")
