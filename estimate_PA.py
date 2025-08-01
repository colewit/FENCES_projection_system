
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import tqdm

def PA_estimator(obs_df, kde, n_neighbors=500, n_total_samples=100000):
    # Step 1: Sample from the KDE joint distribution
    joint_samples = kde.resample(n_total_samples).T  # shape: (n_samples, 4)

    # Step 2: Scale 'rolled_PA' and 'Age' jointly using StandardScaler
    scaler = StandardScaler()
    scaled_joint = scaler.fit_transform(joint_samples[:, :2])  # Only scale first 2 cols

    # Step 3: Fit NearestNeighbors on scaled joint
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(scaled_joint)

    # Step 4: Scale the observation using the same scaler
    obs = obs_df[['rolled_PA', 'Age']].iloc[0].values
    obs_scaled = scaler.transform([obs])

    # Step 5: Find neighbors and sample
    idx = nn_model.kneighbors(obs_scaled, return_distance=False)[0]

    # Step 6: Sample from the PA_next column (column 2 in joint_samples, original space)
    num_samples = obs_df.shape[0]
    sample_PA = np.random.choice(joint_samples[idx, 2], size=num_samples, replace=True)

    return sample_PA


if __name__ == '__main__':

    with open('player_dicts_2020_2024_test.pkl','rb') as f:
        player_dict_list=pickle.load(f)


    data = pd.read_csv('data/cached_data_obj.csv').dropna(subset=['wRC_plus_next'])
    data['wRC_plus_flag'] = np.where(data.wRC_plus_next > 115, 1, np.where(data.wRC_plus_next.between(95, 115), 2, 3)) 

    cols = ['rolled_PA', 'Age', 'PA_next']
    joint_data = data[data.Season <= 2019].dropna(subset=cols)

    # separate KDE's for players who are 'good enough' such that if they are healthy they will likely get fill playing time or not
    X_good = joint_data[joint_data.wRC_plus_flag==1][cols].values.T
    X_okay = joint_data[joint_data.wRC_plus_flag==2][cols].values.T
    X_bad = joint_data[joint_data.wRC_plus_flag==3][cols].values.T
        
    print(X_bad.shape, X_good.shape, X_okay.shape)
    
    kde_good = gaussian_kde(X_good)
    kde_okay = gaussian_kde(X_okay)
    kde_bad = gaussian_kde(X_bad)

    
    with open('models/kde.pkl','wb') as f:
        pickle.dump({1:kde_good, 2:kde_okay, 3:kde_bad}, f)
    '''
    full_player_data = []

    pbar = tqdm.tqdm(player_dict_list)
    for item in pbar:

        dist = item.distribution_params_dict.get('blended_neighbors_params')
        if dist is None:
            continue
            
        skew, loc, scale = dist
        baseline = item.player_row.baseline_wRC_plus

        obs_df = item.get_fast_baseline_projections( (skew, loc, scale ))
        obs_df['wRC_plus_flag'] = np.where(obs_df.wRC_plus > 115, 1, np.where(obs_df.wRC_plus.between(95, 115), 2, 3)) 

        player_df = []
        for wrc_plus_flag, sub_df in obs_df.groupby('wRC_plus_flag'):

            if wrc_plus_flag == 1:
                kde = kde_good
            elif wrc_plus_flag == 2:
                kde =kde_okay
            else:
                kde = kde_bad

            sample_PA = PA_estimator(sub_df, kde, n_neighbors=500, n_total_samples = 100000)
            sub_df['PA'] = (sample_PA).astype(float)

            sub_df['HR'] = sub_df['HR_rate']*sub_df.PA

            player_df.append(sub_df)

        player_df = pd.concat(player_df)
        full_player_data.append(player_df)
        
    full_player_data = pd.concat(full_player_data)
    full_player_data.to_csv('predictions_test.csv')
    '''
        

   