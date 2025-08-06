import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy.stats import norm
import warnings
import pickle

from fences.utils import preprocess
from fences.utils import build_rolling_windows, kernel, find_nearest_neighbors
from fences.utils import fit_skew_agg, fit_skew_weighted, compute_skewnorm_nll

from fences.utils import get_marginal_cdf_values, sample_joint_copula
from fences.utils import invert_skewnorm_joint_samples_individual, predict_xgb


#warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="resource_tracker:.*")

'''

TODO change all Name uses to IDfg (comparison player and Name)
'''



def get_soft_bucket_error(adf):
    """Smooth softmax-like membership across 10 percentile buckets using bucket centers"""
    pcolumns = ['p01', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p99']
    adf = adf.copy()
    
    # Add fake edges to ensure p0 and p100 exist
    adf['p0'] = -100
    adf['p100'] = 100

    # Use p01–p99 for 10 buckets
    bucket_edges = ['p01', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p99']
    percentiles = ['p0'] + bucket_edges + ['p100']
    pvals = adf[percentiles].values  # shape (n, 13)
    deltas = adf['delta'].values

    n = len(deltas)
    weights = np.zeros((n, 10))  # 10 buckets from p01–p99

    for i in range(n):
        delta = deltas[i]
        ps = pvals[i]  # all 13 percentiles for this row

        # Compute centers of 10 buckets: (p1 + p2)/2 through (p10 + p11)/2
        centers = [(ps[j] + ps[j + 1]) / 2 for j in range(1, 11)]  # 10 centers

        # Find closest two centers
        diffs = [abs(delta - c) for c in centers]
        j = np.argmin(diffs)

        if delta <= centers[0]:
            weights[i, 0] = 1.0
        elif delta >= centers[-1]:
            weights[i, -1] = 1.0
        else:
            if delta < centers[j]:
                j1, j2 = j - 1, j
            else:
                j1, j2 = j, j + 1

            # Interpolate linearly between j1 and j2
            c1, c2 = centers[j1], centers[j2]
            if c2 - c1 == 0:
                weights[i, j1] = 0.5
                weights[i, j2] = 0.5
            else:
                frac = (delta - c1) / (c2 - c1)
                weights[i, j1] = 1 - frac
                weights[i, j2] = frac

    buckets =np.nanmean( weights, axis = 0)
    rmse = np.mean((buckets-.1)**2)**.5
    return rmse

def evaluate_player(data, idfg, start_season, n_seasons_forward, weights, tau=0.10, predict_delta = True, sim_num = 0):
    
    target_columns = ['K%','BB%','1B%','2B%','3B%','HR%','out%', 'xwOBA']
    delta_target_columns = [f'delta_{column}' for column in target_columns]
    
    latent_columns = ['K%','chase_value', 'mab_launch_speed', 'mab_launch_angle', 
                      'mab_woba', 'BB%', 'Barrel%', 'whiff', 'xwOBA','single_proba','double_proba', 'triple_proba','home_run_proba']
    
    latent_next_columns = [f'{col}_next' for col in latent_columns]
    #['xBA','xSLG','EV', 'PA', 'wRC_plus']
    pca_columns = list(set(latent_columns + ["PA"]))
    
    
    cdf = data[['Season','IDfg','Name','Age', 'wRC_plus_next', 'rolled_wRC_plus','PA_next'] \
               + list(set(pca_columns + latent_next_columns + target_columns))].copy()

    cdf[[x+'_z_score' for x in pca_columns]] = cdf.groupby('Season')[pca_columns]\
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    
    cdf_means = cdf.groupby('Season')[pca_columns].agg('mean').reset_index()
    cdf_stds = cdf.groupby('Season')[pca_columns].agg('std').reset_index()
    z_scales = cdf_means.merge(cdf_stds, how = 'inner', on = 'Season', suffixes = ['_mean', '_std'])
    
    l=[]
    
    last_row = z_scales[z_scales.Season == z_scales.Season.max()].copy()
    for i in range(1, n_seasons_forward+1):
        last_row['Season'] +=1
        l.append(last_row.copy(deep=True))
    z_scales = pd.concat([pd.concat(l), z_scales])
    
    cdf = cdf[cdf.Season >= 2015]

    n_season_window = 3
    cdf_roll = build_rolling_windows(cdf, pca_columns, n_seasons=n_season_window)
    cdf_roll = cdf_roll.dropna(subset=['xwOBA'])
    
    pdf = (
        cdf[(cdf.IDfg == idfg) & (cdf.Season.between(start_season - n_season_window + 1, start_season))]
        .sort_values('Season', ascending=False)
        .reset_index(drop=True)
    )
    
    for season in range(start_season, start_season+n_seasons_forward, 1):
    
        all_neighbors = find_nearest_neighbors(pdf, cdf, idfg, season, pca_columns, 400, 3, weights, tau, 'exp',10)
        all_neighbors = all_neighbors[['Name', 'IDfg', 'Season', 'PA_next', 'PA', 'neighbor_w'] +\
                                      list(set(latent_columns + latent_next_columns + target_columns)) ]
        

        all_params = [pdf.iloc[0]]
        for column in latent_columns:
            
            all_neighbors[f'delta_{column}_forward'] = all_neighbors[f'{column}_next'] - all_neighbors[column]

            params = fit_skew_agg(all_neighbors, column=column)

            all_params.append(params)
        
            all_neighbors[f'skew_{column}'] = params[f'skew_{column}']
            all_neighbors[f'scale_{column}'] = params[f'scale_{column}']
            all_neighbors[f'loc_{column}'] = params[f'loc_{column}']
            
            all_neighbors = get_marginal_cdf_values(all_neighbors, column)
            
        all_params = pd.concat(all_params)

        # Convert uniform marginals to standard normals
        z_data = all_neighbors[[f'u_{col}' for col in latent_columns]].applymap(norm.ppf).dropna()

        # Estimate correlation matrix
        copula_corr = z_data.corr()
        u_samples = sample_joint_copula(copula_corr, latent_columns, n_samples=1000)
    

        with open('fences/xgb_woba.pkl', 'rb') as f:
            d = pickle.load(f)
            xgb = d['model']
            output_columns = d['target_columns']
            feature_columns = d['feature_columns']

        xgb_columns = latent_columns
        output_df = invert_skewnorm_joint_samples_individual(all_params, u_samples, latent_columns)
        #output_df = predict_xgb(samples, xgb, feature_columns, output_columns)

        last_row = pdf[pdf.Season == pdf.Season.max()][['Season', 'IDfg', 'Name', 'Age']]
    
        sample_row = output_df.sample(1)
        sample_row['Age'] = last_row['Age'].iloc[0] + 1
        sample_row['Season'] = last_row['Season'].iloc[0] + 1
        sample_row['IDfg'] = last_row['IDfg'].iloc[0]
        
        sample_row['PA'] = np.random.normal(500, 75) if sample_row.xwOBA.iloc[0] > .320 else np.random.normal(400, 75)
        
        cols_to_fill = [ 'Barrel%', 'whiff', 
         'K%',  'mab_launch_speed','xwOBA',  'chase_value',  'BB%',
         'mab_launch_angle', 'mab_woba']
        
        pdf = pd.concat([sample_row, pdf])
        pdf = pdf.merge(z_scales, how = 'left', on = 'Season')
        
        pd.options.display.max_columns = 300
        
        for column in pca_columns:
            
            pdf[column+'_z_score'] = (pdf[column] - pdf[f'{column}_mean'])/(pdf[f'{column}_std'] + 1e-6)
            
        pdf.drop(columns = [x for x in z_scales.columns if x != 'Season'], inplace=True)
        
    pdf['sim']=sim_num
    return pdf

if __name__ == '__main__':
    
    PREDICT_DELTA = False
    
    name = 'Elly De La Cruz'
    n_sims = 10
    
    
    season = 2024
    n_seasons_forward = 3
    
    target_columns = ['K%','BB%','1B%','2B%','3B%','HR%','out%', 'xwOBA']
    
    latent_columns = ['K%','chase_value', 'mab_launch_speed', 'mab_launch_angle', 
                      'mab_woba', 'BB%', 'Barrel%', 'whiff', 'xwOBA','single_proba','double_proba', 'triple_proba','home_run_proba']
    
    data = preprocess(latent_columns, target_columns)
    idfg = data[data.Name==name].IDfg.iloc[0]
    
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_player)(data.copy(deep=True), idfg, season, n_seasons_forward, [.72, .14, .14], 16, False, i)
        for i in range(n_sims)
    )

    pd.concat(results).to_csv('samples.csv')
    