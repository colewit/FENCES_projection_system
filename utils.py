
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import skewnorm, norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

NA_FILLS_DEFAULT = {
        'PA': -2, 'xwOBA': -2, 'Barrel%': -1, 'BB%': -1, 'K%': 1, 
        'mab_launch_speed':-1, 'chase_value':-1, 'mab_launch_angle':0, 'mab_woba':-1,
        'whiff':-1, 'home_run_proba':-1, 'double_proba':-1, 'single_proba':-1, 'triple_proba':0
    }
    
def build_rolling_windows(df, columns, n_seasons=3):
    df = df.sort_values(['IDfg', 'Season'])
    out = df.copy()
    for i in range(n_seasons):
        shifted = df.groupby('IDfg')[columns].shift(i)
        shifted.columns = [f'{col}_back_{i}' for col in columns]
        out = pd.concat([out, shifted], axis=1)
    return out.reset_index(drop=True)

def kernel(distance, tau=0.10, kind="exp"):
    if kind == "exp":
        return np.exp(-distance / tau)
    elif kind == "gauss":
        return np.exp(-(distance**2) / (2 * tau**2))
    else:
        raise ValueError("kind must be 'exp' or 'gauss'")

def preprocess(latent_columns, target_columns):
    data= pd.read_csv('data/cached_data_obj.csv')
    data = data[data.Level=='MLB']
    
    df=pd.read_csv('fences/batter_qualities.csv')
    

    df = df.sort_values('Season')
    df['IDfg'] = df['IDfg'].astype(str)

    latent_next_columns = [f'{col}_next' for col in latent_columns]
    
    selected_columns = [x for x in df.columns if x in latent_columns+latent_next_columns]
    data = data.merge(df[['Season', 'num_swings', 'IDfg']+selected_columns], how = 'left', on = ['IDfg','Season'])


    data['in_play%'] = 1 - data['K%'] - data['BB%']
    for col in ['home_run_proba','single_proba','triple_proba','double_proba']:
        data[col] *= data['in_play%']
        
        
    data = data.sort_values('Season')
    for col in ['whiff','Barrel%','BB%','K%','xwOBA', 'home_run_proba','single_proba','triple_proba','double_proba']:
        data[f'{col}_next'] = data.groupby(['IDfg'])[col].transform(lambda x: x.shift(-1))

    data['1B%'] = data['1B']/data.PA
    data['2B%'] = data['2B']/data.PA
    data['3B%'] = data['3B']/data.PA
    data['HR%'] = data['HR']/data.PA
    data['out%'] = 1 - data['OBP']

    for col in list(set(latent_columns + target_columns)):
        data[f'{col}_prev'] = data.groupby('IDfg')[col].transform(lambda x:x.shift(1))
        data[f'delta_{col}'] = data[col] - data[f'{col}_prev']
        
    return data

def find_nearest_neighbors(
    pdf, cdf, idfg, season, columns,
    n_neighbors=None, n_seasons=3,
    weights=None, tau=0.1, kernel_kind="exp",
    noisy_neighbor_cutoff = None, na_fills = NA_FILLS_DEFAULT
):

    if weights is None:
        weights = [1 / n_seasons] * n_seasons
    assert len(weights) == n_seasons

    pdf = (
        pdf[pdf.Season.between(season - n_seasons + 1, season)]
        .sort_values('Season', ascending=False)
        .reset_index(drop=True)
    )
    
    name = pdf.Name.iloc[0]
    
    num_seasons_observed = pdf.shape[0]

    # Filter candidate df, drop rows missing key targets, restrict age window
    cdf_filtered = cdf[cdf.Season >= 2015]
    
    #cdf_filtered.xwOBA_next = cdf_filtered.xwOBA_next.fillna(cdf_filtered.xwOBA - .03)
    cdf_filtered = cdf_filtered.dropna(subset=['xwOBA', 'xwOBA_next'])
    
    cdf_roll = build_rolling_windows(cdf_filtered, columns, n_seasons=n_seasons)
    cdf_roll = cdf_roll.dropna(subset=[f'{columns[0]}_back_0'])  # Require first col back_0 non-na

    age_max = pdf.Age.max()
    cdf_roll = cdf_roll[cdf_roll.Age.between(age_max - 3, age_max + 3)].copy()

    # Predefine fills for missing data (only used during scaling)
    cdf_roll['DistanceToTarget'] = 0.0

    # Create a container for storing feature distances and weights
    for col in columns:
        cdf_roll[f'dist_{col}'] = 0.0
        cdf_roll[f'weight_{col}'] = 0.0

    for i, w in enumerate(weights):
        # For missing seasons beyond observed, copy earliest row and set na columns to np.nan
        if i < num_seasons_observed:
            player_row = pdf.iloc[i].copy()
        else:
            player_row = pdf.iloc[-1].copy()
            for col in na_fills.keys():
                player_row[col] = np.nan

        player_vec = player_row[columns]

        back_columns = [f'{col}_back_{i}' for col in columns]
        X = cdf_roll[back_columns].values

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=back_columns)

        target_z = (player_vec - scaler.mean_) / scaler.scale_
        for col in na_fills.keys():
            col_back = f'{col}_back_{i}'
            if col_back in X_scaled.columns:
                X_scaled[col_back] = X_scaled[col_back].fillna(na_fills[col])
            if np.isnan(target_z[col]):
                target_z[col] = na_fills[col]

        wRC_plus_idx = np.where([x == 'xwOBA' for x in columns])[0][0]
        feat_weights = (((target_z.values) + 1) ** 2).clip(0, 9)
        feat_weights[wRC_plus_idx] = 18

        # Save per-feature distances before PCA
        for j, col in enumerate(columns):
            col_back = f'{col}_back_{i}'
            if col_back not in X_scaled:
                continue
            diffs = np.abs(X_scaled[col_back].values - target_z[col]) * feat_weights[j]
            cdf_roll[f'dist_{col}'] += np.abs(X_scaled[col_back].values - target_z[col]) * w
            cdf_roll[f'weight_{col}'] += w * feat_weights[j]

        # PCA-reduced total distance (for main neighbor distance)
        X_weighted = X_scaled.values * feat_weights
        target_weighted = target_z.values * feat_weights

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_weighted)
        target_pca = pca.transform(target_weighted.reshape(1, -1))

        distances = np.abs(X_pca - target_pca).sum(axis=1) * w
        cdf_roll['DistanceToTarget'] += distances

    # Drop self-comparisons
    cdf_roll = cdf_roll[~np.logical_and(cdf_roll.IDfg == idfg, cdf_roll.Season == season)]

    PA = pdf.iloc[0].PA
    #tau = tau * (np.sqrt(600)/np.sqrt(PA)).clip(1)

    # Apply kernel weighting to total distances
    cdf_roll['neighbor_w'] = kernel(cdf_roll['DistanceToTarget'], tau, kernel_kind)

    # Keep top N neighbors if requested
    if n_neighbors is not None:
        cdf_roll = cdf_roll.nlargest(n_neighbors, 'neighbor_w').copy()
        
    # cutoff all neighbors with
    if noisy_neighbor_cutoff is not None:
        cdf_roll = cdf_roll[cdf_roll.neighbor_w >= (cdf_roll.neighbor_w.max()/noisy_neighbor_cutoff)]

    # Add metadata
    cdf_roll['comparison_season'] = season
    cdf_roll['comparison_player'] = name
    cdf_roll['comparison_IDfg'] = idfg

    return cdf_roll

def fit_skew_weighted(delta, w):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()

  
    def nll(params):
        a, loc, scale = params
        if scale <= 0:
            return np.inf
        ll = skewnorm.logpdf(delta, a, loc=loc, scale=scale)
        return -np.sum(w * ll)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a0, loc0, scale0 = skewnorm.fit(delta)
            res = minimize(nll, x0=[a0, loc0, scale0], method="L-BFGS-B",
                       bounds=[(-15, 15), (None, None), (1e-6, None)])
        return res.x if res.success else (np.nan, np.nan, np.nan)
    except:
        
        import traceback
        traceback.print_exc()
        return (np.nan, np.nan, np.nan)

def fit_skew_agg(df, column):
    
    
    df = df.dropna(subset=[f'delta_{column}_forward'])
    a, loc, scale = fit_skew_weighted(df[f'delta_{column}_forward'].values, df['neighbor_w'].values)
    return pd.Series({f'skew_{column}': a, f'loc_{column}': loc, f'scale_{column}': scale})

def compute_skewnorm_nll(adf, column):
    logpdfs = skewnorm.logpdf(adf[f'delta_{column}'], adf['a_fit'], loc=adf['loc_fit'], scale=adf['scale_fit'])
    logpdfs = np.clip(logpdfs, a_min=-1e10, a_max=None)
    return -np.nanmean(logpdfs)


def get_marginal_cdf_values(df, column):
    loc = df[f"loc_{column}"].iloc[0]
    scale = df[f"scale_{column}"].iloc[0]
    skew = df[f"skew_{column}"].iloc[0]
    
    # Value to transform: delta = actual - predicted
    delta = df[f"delta_{column}_forward"]

    df[f'u_{column}'] = skewnorm.cdf(delta, skew, loc=loc, scale=scale)
    return df
       
def sample_joint_copula(copula_corr, columns, n_samples=1000):
    # Sample from multivariate normal
    mvn_samples = np.random.multivariate_normal(mean=[0]*len(columns), cov=copula_corr, size=n_samples)
    
    # Convert back to uniform
    u_samples = norm.cdf(mvn_samples)
    return u_samples

def invert_skewnorm_joint_samples(df, u_samples, columns):
    
    
    l=[]
    for i in range(df.shape[0]):
        
        row = df.iloc[i]
        pdf = pd.DataFrame()
        for j, col in enumerate(columns):
            skew = row[f"skew_{col}"]
            loc = row[f"loc_{col}"]
            scale = row[f"scale_{col}"]
            samples = skewnorm.ppf(u_samples[:, j], skew, loc=loc, scale=scale)
            pdf[f'delta_{col}_forward'] = samples
            pdf[col] = row[col] + pdf[f'delta_{col}_forward']
        
        pdf['Name'] = row.Name
        pdf['IDfg'] = row.IDfg
        pdf['Season'] = row.Season
        l.append(pdf)
    return pd.concat(l)

def invert_skewnorm_joint_samples_individual(row, u_samples, columns):
    
    
    l=[]
    pdf = pd.DataFrame()
    for j, col in enumerate(columns):
        skew = row[f"skew_{col}"]
        loc = row[f"loc_{col}"]
        scale = row[f"scale_{col}"]
        samples = skewnorm.ppf(u_samples[:, j], skew, loc=loc, scale=scale)
        pdf[f'delta_{col}_forward'] = samples
        pdf[col] = row[col] + pdf[f'delta_{col}_forward']

    pdf['Name'] = row.Name
    pdf['IDfg'] = row.IDfg
    pdf['Season'] = row.Season
    return pdf

def predict_xgb(data, xgb, input_columns, output_columns):
    
    # Columns to predict
    raw_preds = xgb.predict(data[input_columns])   # shape (n_samples, 6)


    # TODO here
    adf = pd.concat([data[['Name','Season', 'IDfg']+input_columns].reset_index(drop=True),
           pd.DataFrame(raw_preds, columns = [f'projected_{col}' for col in output_columns]).reset_index(drop=True)], axis = 1)
    return adf