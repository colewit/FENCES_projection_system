
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
