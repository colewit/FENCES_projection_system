
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm # Statistical distributions and functions
from scipy.interpolate import interp1d # Interpolation functions

from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction techniques
from sklearn.neighbors import NearestNeighbors # Nearest neighbor search
from sklearn.preprocessing import StandardScaler # Data scaling




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
    
class BatterSeasonProfile(object):
        

    def __init__(self, data, player, season, historic_columns, modern_columns, standard_columns):
    
        self.player = player
        self.season = season
        self.player_row = data[(data.Name == player) & (data.Season == season)].iloc[0]
        self.level = self.player_row.Level
        
        d=pd.DataFrame(self.player_row).T
        na_columns = d.isna().sum()
        self.non_na_columns = list(pd.DataFrame(na_columns[na_columns == 0]).T.columns)

        self.historic_columns = [x for x in historic_columns if x in self.non_na_columns]
        self.modern_columns = [x for x in modern_columns if x in self.non_na_columns]
        self.standard_columns = standard_columns

        self.historic_columns_extended = self.historic_columns
        self.historic_columns_extended += ['rolled_'+x for x in self.historic_columns if 'rolled_'+x in self.non_na_columns]
        self.historic_columns_extended += ['rolled_std_'+x for x in self.historic_columns if 'rolled_std_'+x in self.non_na_columns]

        self.modern_columns_extended = self.modern_columns
        self.modern_columns_extended += ['rolled_'+x for x in self.modern_columns if 'rolled_'+x in self.non_na_columns]
        self.modern_columns_extended += ['rolled_std_'+x for x in self.modern_columns if 'rolled_std_'+x in self.non_na_columns]

        self.standard_columns_extended = self.standard_columns
        self.standard_columns_extended += ['rolled_'+x for x in self.standard_columns if 'rolled_'+x in self.non_na_columns]
        self.standard_columns_extended += ['rolled_std_'+x for x in self.standard_columns if 'rolled_std_'+x in self.non_na_columns]
        

        self.full_columns = list(set(self.historic_columns_extended + self.modern_columns_extended + self.standard_columns_extended))

        self.distribution_params_dict = {}
    
    # --- Distribution Building and Projection Functions ---
    
    def build_distribution_for_player(self, data, beta=0.3, alpha=0.5, show_plots=True,
                                      modern_feature_importances=None, historic_feature_importances=None, n_samples=10000):
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
        print(f"Building Distribution for {self.player} {self.season}...")
        data = data.sort_values(['IDfg', 'Season']).copy()

        # Ensure target variable exists (or create if needed, though it should be pre-calculated)
        if 'wRC_plus_next' not in data.columns:
            data['wRC_plus_next'] = data.groupby('IDfg')['wRC_plus'].shift(-1)
    
        # --- Find Similar Players ---
        cdf = self.find_cluster(data) # Find broad cluster based on wRC+ trajectory
    
        # --- Find Nearest Neighbors (Refined Similarity) ---
        # Define number of neighbors based on level (more for MiLB due to higher variance/less data)
        historic_neighbors_n = 50 if self.level == 'MLB' else 100
        modern_neighbors_n = 50 if self.level == 'MLB' else 0 # Only use modern for MLB players
    
        # Initialize DataFrames
        ndf_historic = pd.DataFrame()
        ndf_modern = pd.DataFrame()
        cdf_h = cdf # Base cluster data used for historic search
        cdf_m = cdf # Base cluster data used for modern search (will be filtered by find_neighbors)
    
        ndf_historic, cdf_h, _, _ = self.find_neighbors_for_player(cdf, era='historic',
                                                              n_neighbors=historic_neighbors_n, n_components=5,
                                                              feature_importance=historic_feature_importances)
    
        # Find modern neighbors only if the player is MLB and season allows
        if self.level == 'MLB' and self.season >= 2015 :
            ndf_modern, cdf_m, _, _ = self.find_neighbors_for_player(cdf, era='modern',
                                                                n_neighbors=modern_neighbors_n, n_components=5,
                                                                feature_importance=modern_feature_importances)
    
        # Combine neighbor sets and cluster data used
        ndf = pd.concat([ndf_modern, ndf_historic], ignore_index=True)
        # Use the potentially filtered cluster data from neighbor search steps
        # Combine the dataframes that were *actually used* as input for the neighbor searches
        cdf_combined = pd.concat([cdf_m, cdf_h], ignore_index=True).drop_duplicates(subset=['Name', 'Season', 'IDfg']) # Use IDfg too
    
    
        # --- Analyze Next Season Performance of Comps ---
        # Calculate the change in wRC+ from rolled_wRC+ for comps
        cdf_combined = cdf_combined.dropna(subset=['wRC_plus_next', 'baseline_wRC_plus'])
        ndf = ndf.dropna(subset=['wRC_plus_next', 'baseline_wRC_plus'])
    
    
        # Deltas for the broader cluster
        x_deltas = (cdf_combined['wRC_plus_next'] - cdf_combined['baseline_wRC_plus']).astype(float)
        # Deltas for the nearest neighbors
        y_deltas = (ndf['wRC_plus_next'] - ndf['baseline_wRC_plus']).astype(float)
    
    
        # --- Fit Skew-Normal Distributions ---
        # Fit to broader cluster 
        a_fit1, loc_fit1, scale_fit1 = skewnorm.fit(x_deltas if not x_deltas.empty else y_deltas)
        
        # Fit to nearest neighbors 
        a_fit2, loc_fit2, scale_fit2 = skewnorm.fit(y_deltas if not y_deltas.empty else x_deltas)
    

        self.distribution_params_dict['broad_neighbors_params'] = (a_fit1, loc_fit1, scale_fit1)
        self.distribution_params_dict['broad_neighbors_params_readme'] = ' '.join('''params for fitted skewed normal for change in wRC+ nearest 1000 neighbors based on wRC+ by age pattern. 
                                                                            Params are a, loc, and scale (used like skewnorm.rvs(a, loc=loc, scale=scale))'''.split())

        self.distribution_params_dict['nearest_neighbors_params'] = (a_fit2, loc_fit2, scale_fit2)
        self.distribution_params_dict['nearest_neighbors_params_readme'] =' '.join( '''params for fitted skewed normal for change in wRC+ nearest 100 neighbors based on batting profile. 
                                                                              Params are a, loc, and scale (used like skewnorm.rvs(a, loc=loc, scale=scale))'''.split())
        
        # --- Combine Distributions and Apply Aging Curve ---
        # Blend samples from the two distributions based on beta
        # Beta weights the broader cluster (cdf), (1-beta) weights neighbors (ndf)
        n_intermediate_samples = 10000 # Number of samples for intermediate fitting
        n1 = int(n_intermediate_samples * beta)
        n2 = int(n_intermediate_samples * (1 - beta))
    
        samples1 = skewnorm.rvs(a_fit1, loc=loc_fit1, scale=scale_fit1, size=n1)
        samples2 = skewnorm.rvs(a_fit2, loc=loc_fit2, scale=scale_fit2, size=n2)
        combined_samples = np.concatenate([samples1, samples2])
    
        # Fit a final skew-normal distribution to the combined samples
        a_fit_comb, loc_fit_comb, scale_fit_comb = skewnorm.fit(combined_samples)

        self.distribution_params_dict['blended_beta'] = beta
        self.distribution_params_dict['blended_neighbors_params'] = (a_fit_comb, loc_fit_comb, scale_fit_comb)
        self.distribution_params_dict['blended_neighbors_params_readme'] = ''.join('''params for fitted skewed normal for change in wRC+ blended from broad_neighbors_params and
                                                                              nearest_neighbors_params. Params are a, loc, and scale (used like skewnorm.rvs(a, loc=loc, scale=scale)). 
                                                                              blended_beta is portion of samples to come from broad neighbors and (1-beta) is portion of samples from 
                                                                              nearest neighbors before fitting combined skewnormal'''.split())
        
    
        # Load generic aging curve data
        try:
            aging_curve = pd.read_csv('aging_curve.csv')
            aging_curve_shift = aging_curve.loc[aging_curve.Age == self.player_row.Age, 'delta'].iloc[0]
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
        delta_samples = skewnorm.rvs(*final_params, size=n_samples) # Generate 1000 samples for final output
        # Add the player's rolled wRC+ to get the projected wRC+ distribution
        projected_wrc_samples = delta_samples + self.player_row.baseline_wRC_plus

        self.distribution_params_dict['shifted_beta'] = beta
        self.distribution_params_dict['shifted_alpha'] = beta
        self.distribution_params_dict['shifted_neighbors_params'] = (a_fit_comb, loc_fit_comb, scale_fit_comb)
        self.distribution_params_dict['shifted_neighbors_params_readme'] = ''.join('''params for fitted skewed normal for change in wRC+ blended from broad_neighbors_params and
                                                                              nearest_neighbors_params and then mean is shrunk towards mean change of aging curve.
                                                                              Params are a, loc, and scale (used like skewnorm.rvs(a, loc=loc, scale=scale)). 
                                                                              shifted_beta is portion of samples to come from broad neighbors and (1-beta) is portion of samples from 
                                                                              nearest neighbors before fitting combined skewnormal, alpha adjusts how agressively to shrink towards aging 
                                                                              curve mean'''.split())
        

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
                 plt.hist(x_deltas, bins=30, density=True, alpha=0.3, label=f"Broad Cluster Deltas (n={len(x_deltas)})")
            if not y_deltas.empty:
                 plt.hist(y_deltas, bins=30, density=True, alpha=0.3, label=f"Closest Neighbor Deltas (n={len(y_deltas)})")
    
            pdf_final_delta = skewnorm.pdf(ln_space, *final_params)
            plt.plot(ln_space, pdf_final_delta, label="Projected Change in wRC+ PDF", color='red', linewidth=2)
            plt.title(f"Historical Comparables for {self.player} {self.season} \n Delta wRC+ Distribution (NextYr - EwMA of last 4 seasons)")
            plt.xlabel("wRC+ Change")
            plt.ylabel("Density")
            plt.xlim(max(min_delta, -150), min(max_delta, 150)) # Limit x-axis for readability
            plt.legend(bbox_to_anchor=(0.5, -0.15))
            plt.grid(True, alpha=0.3)
    
    
            # Plot 2: Final Projected wRC+ Distribution
            plt.subplot(1, 2, 2)
            plt.hist(projected_wrc_samples, bins=30, density=True, alpha=0.6, label="Projected wRC+")
            plt.axvline(self.player_row.rolled_wRC_plus, color='blue', linestyle='--', label=f"EwMA of wRC+ \n from {self.season - 4} thru {self.season}  ({self.player_row.rolled_wRC_plus:.0f})")
            if 'wRC_plus_next' in self.player_row and not pd.isna(self.player_row.wRC_plus_next):
                plt.axvline(self.player_row.wRC_plus_next, color='red', linestyle='-', label=f"Actual wRC+ in {self.season+1} ({self.player_row.wRC_plus_next:.0f}) ")

            plt.axvline(x=self.player_row.baseline_wRC_plus, color = 'green', label = f'Baseline wRC+ after {self.season} ({self.player_row.baseline_wRC_plus:.0f})' )
            plt.axvline(x=projected_wrc_samples.mean(), color = 'yellow', label = f'Projected wRC+ for {self.season + 1} ({projected_wrc_samples.mean():.0f})' )

            plt.title(f"{self.player} {self.season+1} - Projected wRC+")
            plt.xlabel("Projected wRC+")
            plt.ylabel("Density")
            # Determine xlim dynamically for projection plot
            min_proj = np.percentile(projected_wrc_samples, .5)
            max_proj = np.percentile(projected_wrc_samples, 99.5)
            plt.xlim(min_proj - 10, max_proj + 10)
            plt.legend(bbox_to_anchor=(0.5, -0.15))
            plt.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        print("Distribution building finished.")
        # Return the original cluster df (cdf) and combined neighbors (ndf)
        return projected_wrc_samples, cdf, ndf, final_params
    
    def find_cluster(self, data, max_matches=1000, weights = [0.06, 0.12, 0.16, 0.25, 0.4]):
    
        '''
        Note that default weights were found using linear regression to find best weights to predict wRC+ based on wRC+ from last 5 years
        '''
    
        num_season =  self.player_row.num_season
            
        age = self.player_row.Age
        age = min(max(age, 18), 41)

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

        player_df = data[np.logical_and(data.Name==self.player, data.Season.between(self.season-4, self.season))].sort_values('Season')
        
        names = adf.Name.unique()
        rdf = data[data.Name.isin(names)]
        rdf = rdf[np.logical_or(np.logical_and(rdf.Name==self.player, rdf.Season==self.season), rdf.Season < self.season)]

        baseline_rolled = player_df.rolled_wRC_plus
        baseline = player_df.wRC_plus
        # Suppress only the specific DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
    
            rdf = rdf.groupby('Name').apply(lambda x: rolled_wrc_distance(x, baseline_rolled, weights = weights)).reset_index(drop=True)
            rdf = rdf.groupby('Name').apply(lambda x: wrc_distance(x, baseline, weights = weights)).reset_index(drop=True)
            
        adf = adf.merge(rdf[['Name','rolled_wRC_plus_distance', 'wRC_plus_distance','Season']], how = 'inner', on = ['Name','Season'])

        adf['distance'] = adf.rolled_wRC_plus_distance + adf.wRC_plus_distance
        return adf.sort_values('distance').head(max_matches)
    
    
    def find_neighbors_for_player(self, df, era, n_neighbors=5, n_components=5, feature_importance=None):
        """
        Finds the nearest neighbors for a target player-season using PCA on statistical features.
    
        Filters data based on 'era' (historic vs. modern Statcast), standardizes features,
        applies PCA, and uses NearestNeighbors to find players with the most similar
        principal component vectors.
    
        Args:
            df (pd.DataFrame): Input DataFrame (typically the output of find_cluster).
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
        print(f"  Finding {era}-era neighbors for {self.player} {self.season}...")
        df = df.copy()
    
        # --- Filter data based on era ---
        if era == 'historic':
            # Exclude seasons with significant Statcast data (post-2014 for MLB)
            # Allow all seasons if target is MiLB (as Statcast isn't reliably available there)
            max_season = 2015 if self.level == 'MLB' else 2030 # Allow future seasons for MiLB sims
            df_filtered = df[np.logical_or(
                (df['Season'] == self.season) & (df['Name'] == self.player),
                df['Season'] < max_season
            )].copy()
            columns_to_use = self.historic_columns_extended + self.standard_columns_extended 
        elif era == 'modern':
            # Require modern data (post-2014)
            if self.season < 2015:
                 print(f"    Skipping modern era - target season {self.season} is before 2015.")
                 return pd.DataFrame(), df, None, None # Not applicable
            df_filtered = df[np.logical_or(
                 (df['Season'] == self.season) & (df['Name'] == self.player),
                 df['Season'] >= 2015
            )].copy()
            columns_to_use = self.modern_columns + self.standard_columns_extended 
        else:
            raise ValueError("era must be 'historic' or 'modern'")


        # Add wRC_plus_prev if available
        prev_cols = ['wRC_plus_prev'] if 'wRC_plus_prev' in df_filtered.columns else []
    
        # Combine and ensure columns exist in the filtered data
        final_feature_cols = [col for col in columns_to_use if col in self.full_columns] # Re-check existence
    
        # --- Prepare Data for PCA ---
        # Drop rows with NaNs in any feature column
        df_clean = df_filtered.dropna(subset=final_feature_cols).copy()
        df_clean = df_clean.reset_index(drop=True) # Reset index after dropping rows

        # Find the index of the target player in the cleaned data
        target_idx_arr = df_clean[
            (df_clean['Name'] == self.player) & (df_clean['Season'] == self.season)
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

        n_neighbors = min(n_neighbors, X.shape[0]-1)
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
    
    
    def get_baseline_projections(self, data, similar_players_df, dist_params, n_samples=1000):
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
        print(f"Generating Projections for {self.player} {self.season}")
    
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
        pdf = data[(data.Name == self.player) & (data.Season == self.season)]
    
        # Create a dictionary to store projection results for each sample
        projection_dict = {}
        # Get baseline rolled values from the player's data for the target season
        projection_dict['rolled_wRC_plus'] = float(self.player_row['rolled_wRC_plus']) # Store baseline
        projection_dict['baseline_wRC_plus'] = float(self.player_row['baseline_wRC_plus']) # Store baseline
        projection_dict['wRC_plus_label'] = float(self.player_row['wRC_plus_next']) # Store baseline
        projection_dict['wRC_plus'] = float(self.player_row['rolled_wRC_plus']) + delta_wrc # Add simulated change
        projection_dict['hit_rate'] = float(self.player_row['rolled_hit_rate']) + delta_avg # Add simulated change
        projection_dict['HR_rate'] = float(self.player_row['rolled_HR_rate']) + delta_hr # Add simulated change
        projection_dict['2B_rate'] = float(self.player_row['rolled_2B_rate']) + delta_2b # Add simulated change
        # Assume 3B rate change is 0 (wasn't in correlation matrix) - use raw rate from season T as baseline
        projection_dict['3B_rate'] = float((self.player_row['3B'] if not np.isnan(self.player_row['3B']) else 0) / self.player_row.PA) + 0
        # Calculate projected 1B rate by subtraction
        projection_dict['1B_rate'] = projection_dict['hit_rate'] - projection_dict['HR_rate'] - projection_dict['3B_rate'] - projection_dict['2B_rate']
    
        # Project OBP by adding simulated change to baseline
        projection_dict['OBP'] = float(self.player_row['rolled_OBP']) + delta_obp
    
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
        projection_df['Name'] = self.player
        projection_df['Season'] = self.season + 1 # Projection is for the *next* season
        projection_df['Sample'] = range(1, 1 + len(projection_df)) # Add sample number (1 to n_samples)
    
        # Select and order columns, round numeric columns to 3 decimal places
        projection_df = projection_df[['Name', 'Season', 'Sample'] + [x for x in projection_df if x not in ['Name', 'Season', 'Sample']]].round(3)
    
        return projection_df
