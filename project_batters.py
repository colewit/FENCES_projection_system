
import numpy as np
import pandas as pd
import warnings
import pickle
from joblib import Parallel, delayed

from scipy.stats import norm

from fences.utils import preprocess
from fences.utils import build_rolling_windows, kernel, find_nearest_neighbors
from fences.utils import fit_skew_agg, fit_skew_weighted, compute_skewnorm_nll

from fences.utils import get_marginal_cdf_values, sample_joint_copula
from fences.utils import invert_skewnorm_joint_samples, predict_xgb


warnings.filterwarnings("ignore")


'''

TODO change all Name uses to IDfg (comparison player and Name)

Known issues: undercuts doubles and singles seemingly for guys at the extremes (Arraez, Durran)
and overshoots homeruns for guys who do not homer
'''

def find_nearest_neighbors_wrapper(
    cdf, idfg, season, columns,
    n_neighbors=None, n_seasons=3,
    weights=None, tau=0.1, kernel_kind="exp",
    noisy_neighbor_cutoff = None
):
    
    na_fills = {
        'PA': -2, 'wRC_plus': -2, 'xwOBA': -2, 'Barrel%': -1, 'BB%': -1, 'K%': 1, 
        'mab_launch_speed':-1, 'chase_value':-1, 'mab_launch_angle':0, 'mab_woba':-1,
        'whiff':-1, 'home_run_proba':-1, 'double_proba':-1, 'single_proba':-1, 'triple_proba':0
    }
    
    pdf = (
        cdf[(cdf.IDfg == idfg) & (cdf.Season.between(season - n_seasons + 1, season))]
        .sort_values('Season', ascending=False)
        .reset_index(drop=True)
    )
    
    return find_nearest_neighbors(pdf, cdf, idfg, season, columns, n_neighbors, n_seasons, 
                           weights, tau, kernel_kind, noisy_neighbor_cutoff, na_fills)
    


def project_batters(season, weights, n_neighbors=400, n_seaon_lookback=3, tau=0.10,
                    noisy_neighbor_cutoff=10, predict_delta = True, n_samples = 1000):
    
    target_columns = ['K%','BB%','1B%','2B%','3B%','HR%','out%', 'xwOBA']
    delta_target_columns = [f'delta_{column}' for column in target_columns]
    
    # TODO convert probas from BB rate to rate per PA, and instead of xgboost to predict go from probas
    # tenatively done ^^
    latent_columns = ['K%','chase_value', 'mab_launch_speed', 'mab_launch_angle', 
                      'mab_woba', 'BB%', 'Barrel%', 'whiff', 'xwOBA','single_proba','double_proba', 'triple_proba','home_run_proba']
    
    latent_next_columns = [f'{col}_next' for col in latent_columns]
    pca_columns = list(set(latent_columns + ['PA', 'wRC_plus']))
    
    data = preprocess(latent_columns, target_columns)
    cdf = data[['Season','IDfg','Name','Age', 'wRC_plus_next', 'rolled_wRC_plus','PA_next'] \
               + list(set(pca_columns + latent_next_columns + target_columns))].copy()

    cdf[[x+'_z_score' for x in pca_columns]] = cdf.groupby('Season')[pca_columns]\
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    #TODO make PA a config
    PA_min_cutoff = 200
    
    cdf = cdf[cdf.Season >= 2015]
    names = cdf[np.logical_and(cdf.PA>PA_min_cutoff,cdf.Season == season)].Name.unique()

    cdf_roll = build_rolling_windows(cdf, pca_columns, n_seasons=3)
    cdf_roll = cdf_roll.dropna(subset=['xwOBA'])
    
    combos = cdf_roll[cdf_roll.Season == season][['IDfg','Season']].values
    

    
    
    results = Parallel(n_jobs=-1)(
        delayed(find_nearest_neighbors_wrapper)(cdf, idfg, season,
                                                pca_columns, n_neighbors, n_seaon_lookback,
                                                weights, tau, 'exp',noisy_neighbor_cutoff)
        for idfg, season in combos
    )
    '''
    
    for name, season in combos[:3]:
    
        find_nearest_neighbors(cdf, idfg, season, pca_columns, 400, 3, weights, tau, 'exp',10)
    '''
    
    all_neighbors = pd.concat(results)
    all_neighbors = all_neighbors[all_neighbors.PA_next > PA_min_cutoff]
    
    all_neighbor_params = cdf_roll[['Name','IDfg', 'Season', 'PA_next', 'PA'] + list(set(latent_columns + target_columns)) ]
    all_neighbor_params = all_neighbor_params[all_neighbor_params.Season==season]
    
    
    for column in latent_columns:
        all_neighbors[f'delta_{column}_forward'] = all_neighbors[f'{column}_next'] - all_neighbors[column]
        
        sub_params = (
            all_neighbors.groupby(['comparison_IDfg','comparison_season'])
                         .apply(fit_skew_agg, column = column).reset_index()
                         .rename(columns={'comparison_IDfg':'IDfg','comparison_season':'Season'})
        )
        
        
        all_neighbor_params = all_neighbor_params.merge(
            sub_params,
            how='left', on=['IDfg','Season'])

    all_neighbor_params = all_neighbor_params.dropna(subset=['loc_chase_value'])

    '''
    Now we have the marginals for each of chase value, launch angle, and launch speed.
    We need to make copulas, draw from them, and predict wRC plus for each observed outcome based on drawn swing qualities
    '''

    all_neighbors = all_neighbors.merge(all_neighbor_params\
                                        .rename(columns={'IDfg':'comparison_IDfg','Season':'comparison_season'}),
                                       how = 'left', on = ['comparison_IDfg','comparison_season']
                                       )
    
    for col in latent_columns:
        all_neighbors = all_neighbors\
            .groupby(['comparison_player','comparison_season'])\
            .apply(lambda x: get_marginal_cdf_values(x, col))

    
    # Convert uniform marginals to standard normals
    z_data = all_neighbors[[f'u_{col}' for col in latent_columns]].applymap(norm.ppf).dropna()

    # Estimate correlation matrix
    copula_corr = z_data.corr()
    u_samples = sample_joint_copula(copula_corr, latent_columns, n_samples)
    
    output_df = invert_skewnorm_joint_samples(all_neighbor_params, u_samples, latent_columns)
    
    
    # commenting out because I would prefer to ue samples from dist as predictions.
    # might use a model to calibrate but depends on accuracy - complexity tradeoff.
    '''
    with open('fences/xgb_woba.pkl', 'rb') as f:
        d = pickle.load(f)
        xgb = d['model']
        output_columns = d['target_columns']
        feature_columns = d['feature_columns']

    xgb_columns = latent_columns
    samples = invert_skewnorm_joint_samples(all_neighbor_params, u_samples, latent_columns)
    preds = predict_xgb(samples, xgb, feature_columns, output_columns)
    output_df = preds
    '''
    
    return output_df

if __name__ == '__main__':
    
    PREDICT_DELTA = False
    season = 2024
    sample_df = project_batters(season = season, weights=  [.72,.14,.14], n_seaon_lookback=3,
                                n_neighbors = 400, tau=16, predict_delta = PREDICT_DELTA)
    
    sample_df.to_csv(f'samples_{season}.csv')