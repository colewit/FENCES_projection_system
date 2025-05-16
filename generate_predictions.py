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

# local module imports
import config
from model_minor_league_conversions import get_minor_league_data
from batter_season_profile import BatterSeasonProfile

# --- Data Preparation Functions ---

def prepare_data(major_league_path='../data.csv',minor_league_path='minor_league_data.csv', use_minor_league_model=True):
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
    data = pd.read_csv(major_league_path).rename(columns={'wRC+': 'wRC_plus'})
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
        minor_league_df = pd.read_csv(minor_league_path)
        amdf = get_minor_league_data(minor_league_df = minor_league_df,
                                     major_league_df = data, 
                                     use_model=use_minor_league_model,
                                     fit_model=False) # Fit model should likely be False unless retraining here
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
    #data[f'PA_for_{col}'] = np.where(data[col].isna(), np.nan, data[weight_col])

    # Calculate EWMA of the weights (denominator)
    # Group by player (IDfg) and apply EWM along the season axis
    rolled_pa_col = data[f'rolled_{weight_col}']

    # Calculate the product of the stat and its weight
    data[f'{col}_x_PA'] = data[col] * data[weight_col] # Will be NaN if either is NaN

    # Calculate EWMA of the stat*weight product (numerator)
    rolled_stat_pa_col = data.groupby('IDfg')[f'{col}_x_PA']\
                             .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())

    # Calculate the final exponentially weighted mean
    data[f'rolled_{col}'] = rolled_stat_pa_col / rolled_pa_col.replace(0, np.nan) # Avoid division by zero

    # Calculate EWMA Standard Deviation (more complex)
    # Formula involves EWMA of squared values and EWMA of values
    # EW Var(X) = E[X^2] - (E[X])^2
    data[f'{col}_sq_x_PA'] = (data[col]**2) * data[weight_col]
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
    standard_columns = ['OBP','AVG','SLG', 'hit_rate']
    both_era_columns = ['wRC_plus', 'HR_rate', 'ISO+', '2B_rate']
    historic_columns = ['BB%+', 'K%+', 'AVG+', 'OBP+', 'SLG+', 'BB/K', 'wOBA']
    modern_columns = ['xBA', 'xSLG', 'xwOBA', 'Barrel%', 'Contact%', 'SwStr%', 'maxEV', 'EV', 'LA'] # Statcast era

    processed_columns_tracker = [] # Track generated columns

    data['rolled_PA'] = data.groupby('IDfg')['PA']\
                        .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())
    data['rolled_mlb_PA'] = data.groupby('IDfg')['mlb_PA']\
                        .transform(lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean())
        
    # Process historic and general columns using 'PA' as weight
    cols_to_process_pa = list(set(historic_columns + both_era_columns + standard_columns))
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

    both_era_columns.append('baseline_wRC_plus')
    both_era_columns.append('rolled_PA')
    both_era_columns.append('rolled_mlb_PA')
    
    print("Rolling statistics calculation finished.")
    # Return the dataframe and the lists of base columns *actually used*
    return data, list(set(historic_columns + both_era_columns)), list(set(modern_columns + both_era_columns)), standard_columns



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
        
        player, season, data, historic_columns, modern_columns, standard_columns = args
        player_obj = BatterSeasonProfile(data, player, season, historic_columns, modern_columns, standard_columns)
        samples, cluster_df, neighbors_df, dist_params = player_obj.build_distribution_for_player(data, show_plots=False, beta=config.BETA, alpha=config.ALPHA)

        projection_df = player_obj.get_baseline_projections(data, cluster_df, dist_params, n_samples=1000)
        return player_obj, projection_df
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        print("ERROR FOR", player, season)
        
        return player_obj, pd.DataFrame()

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


def save_player_dict(players_dict_list, path):
    print('Saving BatterSeasonProfile Objects...')
    with open(path,'wb') as f:
        pickle.dump(players_dict_list, f)
        
# --- Main Execution Block ---

if __name__ == '__main__':

        # --- Define Players to Project ---
    # Example: Select players active in 2024 with more than 50 PA
    # Adjust the filtering criteria as needed
    
    min_pa_threshold = config.MIN_PA

    
    chunk_size = config.CHUNK_SIZE      # Number of players to process in each parallel batch
    output_file = config.OUTPUT_DATA_FILE
    start_year = config.START_SEASON
    end_year = config.END_SEASON

    
    # --- Data Loading and Preprocessing ---
    if config.LOAD_FROM_CACHE:
        print("Loading data and columns from cache...")
        
        # Load pre-processed data
        data = pd.read_csv('cached_data_obj.csv')
        # Load pre-calculated column lists
        with open('columns.pkl', 'rb') as f:
            column_data = pickle.load(f)

            historic_columns = column_data['historic'] 
            modern_columns = column_data['modern']
            standard_columns = column_data['standard']
    else:
        print("Processing data from scratch...")
        # Prepare the data (loads raw data, integrates MiLB, etc.)
        # Ensure paths inside prepare_data are correct relative to execution location
        data = prepare_data(major_league_path=config.INPUT_DATA_FILE) # Pass path explicitly

        # Calculate rolling statistics and get column lists
        data, historic_columns, modern_columns, standard_columns = get_rolling_columns(data)

        with open('gam_shrinkage_model.pkl', 'rb') as f:
            gam = pickle.load(f)

        data['baseline_wRC_plus'] = gam.predict(data[['rolled_wRC_plus','rolled_PA']])

        # Save processed data and column lists to cache if overwrite is enabled
        if config.OVERWRITE:
            print("Saving processed data and columns to cache...")

            # Save column lists
            column_data_to_save = {'historic': historic_columns, 'modern': modern_columns, 'standard':standard_columns}
            with open('columns.pkl', 'wb') as f:
                pickle.dump(column_data_to_save, f)
            # Save processed data
            data.to_csv('cached_data_obj.csv', index=False)
            print("Cached data saved.")


    data = data.sort_values(['Season','Name'])
    data['num_season'] = np.where(data.Level=='MiLB', 0, data.num_season)

    print(f"Selecting players from {start_year} to {end_year} with > {min_pa_threshold} PA...")
    season_data = data[(data.PA > min_pa_threshold) & (data.Season.between(start_year, end_year))].copy().sort_values(['Season','Name'])
    
    # Create list of (player_name, season) tuples to process
    # Ensure no duplicates
    player_seasons = list(season_data[['Name', 'Season']].drop_duplicates().itertuples(index=False, name=None))
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

    players_dict_list = []

    
    for i in tqdm.tqdm(range(0, len(player_seasons), chunk_size), desc="Processing Chunks"):
        # Get the current chunk of player-season tuples

        chunk = [(player, season, data, historic_columns, modern_columns, standard_columns) for (player, season) in player_seasons[i:i+chunk_size]]


        # Use ProcessPoolExecutor for parallel execution
        # 'with' statement ensures resources are properly managed
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # map applies process_player to each item in chunk concurrently
            # list() collects the results (order is preserved)
            results_chunk = list(executor.map(process_player, chunk))

        player_chunk = [x[0] for x in results_chunk]
        projection_chunk = [x[1] for x in results_chunk]
        
        # Save the results of the current chunk
        print(f"\nSaving results for chunk {i//chunk_size + 1}...")
        # Write header only for the very first chunk (i == 0)
        save_chunk(projection_chunk, mode='a', header=(i == 0))
        all_results.extend(projection_chunk) # Optional: append to master list

        players_dict_list.extend(player_chunk)
        save_player_dict(players_dict_list, path = f"player_dicts_{start_year}_{end_year}.pkl")

    print(f"\nParallel processing complete. Results saved to {output_file}")
