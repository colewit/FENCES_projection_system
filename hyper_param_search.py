
import pickle
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from tqdm import tqdm
from generate_predictions import PlayerDistribution

if __name__ == '__main__':

    PLAYER_DICTS_FILE = 'player_dicts_2020_2024.pkl'
    start_year = PLAYER_DICTS_FILE

    with open(PLAYER_DICTS_FILE,'rb') as f:
        player_list = pickle.load(f)

    aging_curve = pd.read_csv('aging_curve.csv')
    
    print(len(player_list))

    lowest_error = np.inf
    best_beta = np.nan

    pbar = tqdm([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    for beta in pbar:
        
        means=[]
        actuals = []
        for player in player_list:
            if player is None:
                continue
            a1, loc1, scale1 = player.distribution_params_dict['broad_neighbors_params']
            a2, loc2, scale2 = player.distribution_params_dict['nearest_neighbors_params']

        
            # --- Combine Distributions and Apply Aging Curve ---
            # Blend samples from the two distributions based on beta
            # Beta weights the broader cluster (cdf), (1-beta) weights neighbors (ndf)
            n_total_samples = 10000 # Number of samples for intermediate fitting
 
            n1 = int(n_total_samples * beta)
            n2 = int(n_total_samples * (1 - beta))
        
            samples1 = skewnorm.rvs(a1, loc=loc1, scale=scale1, size=n1)
            samples2 = skewnorm.rvs(a1, loc=loc2, scale=scale2, size=n2)
            combined_samples = np.concatenate([samples1, samples2])
        
            # Fit a final skew-normal distribution to the combined samples
            a_fit_comb, loc_fit_comb, scale_fit_comb = skewnorm.fit(combined_samples)

            samples = skewnorm.rvs(a_fit_comb, loc=loc_fit_comb, scale=scale_fit_comb, size=1000)
            player_mean = samples.mean()
            means.append(player_mean + player.player_row.rolled_wRC_plus)
            actual = player.player_row.wRC_plus_next
            actuals.append(actual)

        error = np.nanmean(abs(np.array(means) - np.array(actuals)))
        
        if error < lowest_error:
            best_beta = beta
            lowest_error = error

        pbar.set_description(f"For beta {beta}, mean error is {error},lowest error is {lowest_error}, best beta is {best_beta}")

    lowest_error = np.inf
    best_alpha = np.nan

    pbar = tqdm([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    for alpha in pbar:
        
        means=[]
        actuals = []
        
        for player in player_list:

            if player is None:
                continue
            a1, loc1, scale1 = player.distribution_params_dict['broad_neighbors_params']
            a2, loc2, scale2 = player.distribution_params_dict['nearest_neighbors_params']

        
            # --- Combine Distributions and Apply Aging Curve ---
            # Blend samples from the two distributions based on beta
            # Beta weights the broader cluster (cdf), (1-beta) weights neighbors (ndf)
            n_total_samples = 10000 # Number of samples for intermediate fitting
 
            n1 = int(n_total_samples * best_beta)
            n2 = int(n_total_samples * (1 - best_beta))
        
            samples1 = skewnorm.rvs(a1, loc=loc1, scale=scale1, size=n1)
            samples2 = skewnorm.rvs(a1, loc=loc2, scale=scale2, size=n2)
            combined_samples = np.concatenate([samples1, samples2])
        
            # Fit a final skew-normal distribution to the combined samples
            a_fit_comb, loc_fit_comb, scale_fit_comb = skewnorm.fit(combined_samples)
            
            age = min(max(player.player_row.Age, 18), 40)
            aging_curve_shift = aging_curve.loc[aging_curve.Age == age, 'delta'].iloc[0]

        
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
            samples = skewnorm.rvs(*final_params, size=1000) # Generate 1000 samples for final output

            player_mean = samples.mean()

            means.append(player_mean + player.player_row.rolled_wRC_plus)
            actual = player.player_row.wRC_plus_next
            actuals.append(actual)

        error = np.nanmean(abs(np.array(means) - np.array(actuals)))

        if error < lowest_error:
            best_alpha = alpha
            lowest_error = error

        pbar.set_description(f"For alpha {alpha}, mean error is {error},lowest error is {lowest_error}, best alpha is {best_alpha}")

    param_dict = {'alpha':best_alpha, 'beta':best_beta}
    
    print(param_dict)
    with open(f'best_alpha_and_beta_params_{start_year}_{end_year}.pkl', 'wb') as f:
        pickle.dump(param_dict, f)
            