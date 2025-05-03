import pandas as pd
import numpy as np
import copy
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import skewnorm, norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
import pickle
import tqdm
import pandas as pd
import numpy as np
import copy
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import skewnorm, norm
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d
import pickle

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


def prepare_data(path = '../data.csv', use_minor_league_model=True):
    
    data = pd.read_csv(path).rename(columns={'wRC+':'wRC_plus'})
    data = data.sort_values('Season')
    
    data['num_season'] = data.groupby('IDfg').Age.transform(lambda x: range(len(x)))

    data['hit_rate'] = (data['HR']+data['3B']+data['2B']+data['1B'])/data['PA']
    data['HR_rate'] = data.HR/data.PA
    data['1B_rate'] = data['1B']/data.PA
    data['2B_rate'] = data['2B']/data.PA
    data['3B_rate'] = data['3B']/data.PA

    
    data['HR_rate_next'] = data.groupby(['IDfg']).HR_rate.transform(lambda x:x.shift(-1))
    data['1B_rate_next'] = data.groupby(['IDfg'])['1B_rate'].transform(lambda x:x.shift(-1))
    data['2B_rate_next'] = data.groupby(['IDfg'])['2B_rate'].transform(lambda x:x.shift(-1))
    data['3B_rate_next'] = data.groupby(['IDfg'])['3B_rate'].transform(lambda x:x.shift(-1))
    data['hit_rate_next'] = data.groupby(['IDfg'])['hit_rate'].transform(lambda x:x.shift(-1))
    
    data['OBP_next'] = data.groupby(['IDfg'])['OBP'].transform(lambda x:x.shift(-1))
    
    data['wRC_plus_next'] = data.groupby('IDfg').wRC_plus.shift(-1)
    data['wRC_plus_prev'] = data.groupby('IDfg').wRC_plus.shift(1)
    

    # change to rolled

    data['PA_next'] = data.groupby('IDfg').PA.shift(-1)
    
    bio_data = pd.read_csv('../People.csv',encoding='latin-1')
    bio_data['Name'] = bio_data.nameFirst+' '+bio_data.nameLast
    bio_data = bio_data[['Name','weight','height','birthYear']].dropna().sort_values('birthYear')

    player_map = pd.read_csv('player_map.csv')
    data = data.merge(player_map, how = 'left', on = 'IDfg')
    data = data.sort_values('birthYear')
    
    data = data.merge( bio_data,how='left', on=['Name','birthYear'])
    data['mlb_PA'] = data['PA']

    amdf = get_minor_league_data(data, use_model = use_minor_league_model, fit_model=False)
    pred_cols = [x for x in amdf.columns if 'pred_' in x]


    amdf = amdf[pred_cols+['Season','Name', 'Age', 'PA','AB', 'IDfg']].rename(columns = {x: x.replace('pred_','') for x in pred_cols})
    amdf["milb_PA"] = amdf.PA
    data['rounded_wRC_plus'] = 2*np.round(data.wRC_plus/2)

    amdf['rounded_wRC_plus'] = 2*np.round(amdf.wRC_plus/2)
    
    
    # map to context adjusted percentiles
    for col in ['BB%','AVG','OBP','SLG']:
        data[f'rounded_{col}'] = data[col].round(2)
        amdf[f'rounded_{col}'] = amdf[col].round(2)
        plus_map = data[['rounded_'+col, col+'+', 'Season']].groupby(['Season','rounded_'+col]).agg('mean').round(0).reset_index()
        amdf = pd.merge_asof(amdf.sort_values(f'rounded_{col}'),plus_map.sort_values(f'rounded_{col}'), on=f'rounded_{col}', by = 'Season')
        pred_cols.append(f'{col}+')
    
    woba_map = data[['rounded_wRC_plus','wOBA', 'Season']].groupby(['Season','rounded_wRC_plus']).agg('mean').reset_index()
    amdf = pd.merge_asof(amdf.sort_values('rounded_wRC_plus'),woba_map[['rounded_wRC_plus','wOBA','Season']].sort_values('rounded_wRC_plus'), on='rounded_wRC_plus', by = 'Season')
    pred_cols.append(f'wOBA')
    
    data['Level'] = 'MLB'
    amdf['Level'] = 'MiLB'
    
    data = pd.concat([amdf, data])
    data = data.reset_index(drop=True)

    # account for uncertainty of MiLB data compared to MLB
    data['PA'] = np.where(data.Level=='MLB',data.PA, data.PA/2)
    data['AB'] = np.where(data.Level=='MLB',data.AB, data.AB/2)
    
    data['multiple_levels'] = data.groupby(['Season', 'IDfg']).Level.transform(len)
    data['Level'] = np.where(data.mlb_PA > 100, 'MLB', np.where(data.mlb_PA.fillna(0) < data.milb_PA.fillna(1),'MiLB','MLB'))

    data_to_dedupe = data[data.multiple_levels>1]

    wm = lambda x: np.average(x, weights=data_to_dedupe.loc[x.index, "PA"])
    

    
    agg_columns = [x.replace('pred_','') for x in pred_cols]
    agg_dict = {x:wm for x in agg_columns}
    agg_dict['PA'] = 'sum'
    agg_dict['AB'] = 'sum'

    # Now, group by 'Name' and 'Season' and calculate the weighted average for each group
    weighted_avg = data_to_dedupe.groupby(['IDfg', 'Season']).agg(agg_dict)
    weighted_avg=weighted_avg.reset_index()
    # Merge the results back to the original data
    
    
    data_to_dedupe = data_to_dedupe.drop(columns=agg_columns+['PA','AB']).merge(weighted_avg[['IDfg', 'Season','PA', 'AB']+agg_columns], on=['IDfg', 'Season'], how='left')
    

    data_unique = data[data.multiple_levels==1]

    data = pd.concat([data_to_dedupe, data_unique])
    data.sort_values('Season')

    data = data.sort_values('Level').groupby(['Name','Season']).first().reset_index()
    data['IDfg'] = data.groupby(['Name'])['IDfg'].transform(lambda x: x.ffill().bfill())
    
    data['composite_id'] = data['Name']+data['birthYear'].astype(str)
    data['IDfg'] = data['IDfg'].combine_first(data['composite_id'])
    return data


def get_minor_league_data(data, path='minor_league_data.csv', shrinkage_PA = 300, use_model=True, 
                          target_cols = ['wRC_plus', 'hit_rate', 'OBP', 'HR_rate', '2B_rate', 'SLG'],
                          fit_model=False, model_path = 'minor_league_model.pkl',  overwrite=False, minimum_pa=20):
    
    minor_league_df = pd.read_csv(path).rename(columns={'PlayerId':'IDfg'})


    if use_model:

        minor_league_df = model_minor_league_conversions(minor_league_df, data, minimum_pa=minimum_pa, 
                                          target_cols = target_cols,
                                          fit_model=fit_model, model_path = model_path, overwrite=overwrite)
        
    else:
        agg_conversion_data = agg_minor_league_conversions(minor_league_df, data, minimum_pa = 100)
        minor_league_df = minor_league_df.merge(agg_conversion_data, how = 'left', on=['Level','League'])
        minor_league_df['wRC_plus_adjusted'] = (minor_league_df['wRC_plus'] * minor_league_df.factor).clip(0,150)
        minor_league_df['Age_for_level'] = minor_league_df.Age - minor_league_df.groupby('Level').Age.transform('median') 
    
        
        # heuristic to give more credit for players playing young for a level. TODO ground empirically
        minor_league_df['age_factor'] = (.97)**(minor_league_df.Age_for_level + 4)
        minor_league_df['pred_wRC_plus'] = (minor_league_df['wRC_plus_adjusted'] * minor_league_df['age_factor'])
        #amdf['pred_wRC_plus'] = (amdf.factor*100 * (shrinkage_PA-amdf.PA).clip(0,shrinkage_PA) + (amdf.PA).clip(0,shrinkage_PA) * amdf['pred_wRC_plus'])/shrinkage_PA
    
    wm = lambda x: np.average(x, weights=minor_league_df.loc[x.index, "PA"])

    agg_dict = {x:wm for x in [y for y in minor_league_df.columns if 'pred' in y] }
    agg_dict['PA'] = 'sum'
    agg_dict['AB'] = 'sum'
    agg_dict['Age'] = 'first'
    amdf = minor_league_df.groupby(['Name','Season','IDfg']).agg(agg_dict).reset_index()
    
        
    
    return amdf
    
def agg_minor_league_conversions(minor_league_df, data, minimum_pa = 100):
    mdf = minor_league_df[minor_league_df.PA>minimum_pa]
    conversion_data = mdf.merge(data[data.PA>minimum_pa][['wRC_plus','Name','Season']].rename(columns = {'wRC_plus': 'mlb_wRC_plus'}), how='left',on=['Name','Season'])
    conversion_data = conversion_data.merge(mdf[mdf.Level=='AAA'][['wRC_plus', 'League','Name','Season']].rename(columns = {'wRC_plus': 'AAA_wRC_plus', 'League':'AAA_league'}), how='left',on=['Name','Season'])
    conversion_data['AAA_wRC_plus'] = np.where(conversion_data.Level=='AAA', np.nan, conversion_data['AAA_wRC_plus'])
    
    conversion_data['factor'] = np.where(conversion_data.Level=='AA', conversion_data['AAA_wRC_plus']/conversion_data['wRC_plus'],conversion_data['mlb_wRC_plus']/conversion_data['wRC_plus'])
    

    agg_conversion_data = conversion_data.groupby(['Level','League']).agg({'factor':'mean'}).reset_index()
    agg_conversion_data['factor'] = np.where(agg_conversion_data.Level=='AA', agg_conversion_data.factor * (agg_conversion_data[agg_conversion_data.Level=='AAA'].factor.mean()),
                                             agg_conversion_data.factor).round(2)
    
    
    return agg_conversion_data



def model_minor_league_conversions(minor_league_df, major_league_df, minimum_pa=400, target_cols = ['wRC_plus', 'hit_rate', 'OBP', 'HR_rate', '2B_rate', 'SLG'],
                                   fit_model=False, model_path='minor_league_model.pkl', overwrite=False):
    
    minor_league_df['Age_for_level'] = minor_league_df.Age - minor_league_df.groupby('Level').Age.transform('mean')


    
    
    # Feature columns
    feature_cols = list(set([
    'League', 'PA','HR_rate','2B_rate', 'BB%', 'K%', 'BB/K', 'AVG','BABIP',
    'OBP', 'SLG', 'OPS', 'ISO', 'Age_for_level', 'wRC_plus', 'hit_rate', 'years_at_level','Spd'
    ]))
    
    
    # Add age-for-level adjustment
    minor_league_df['Age_for_level'] = minor_league_df.Age - minor_league_df.groupby('Level').Age.transform('mean')
    
    minor_league_df['years_at_level'] = minor_league_df.groupby(['IDfg','Level']).Age.transform(lambda x:range(1,len(x)+1))
    
    minor_league_df.IDfg = minor_league_df.IDfg.astype(str)
    major_league_df.IDfg = major_league_df.IDfg.astype(str)
    
    AAA = minor_league_df[minor_league_df.Level=='AAA']
    AA = minor_league_df[minor_league_df.Level=='AA']

    
    promotion_data = AAA.merge(AA[['Season','IDfg', 'Name','Age']+feature_cols], how = 'outer', on = ['Season','IDfg','Name','Age'], suffixes = ['_AAA','_AA'])
    promotion_data['Level']=promotion_data['Level'].fillna('AA')
    
    
    mlb_stats = major_league_df.sort_values('Season')
    mlb_stats['rolling_PA'] = mlb_stats.groupby('IDfg').PA.transform(lambda x:x.iloc[::-1].rolling(window=2, min_periods=1).sum().iloc[::-1])
    
    for col in target_cols:
        mlb_stats[f'PA_x_{col}'] = mlb_stats.PA*mlb_stats[col]
    
        mlb_stats[f'rolling_PA_x_{col}'] = mlb_stats.groupby('IDfg')[f'PA_x_{col}'].transform(lambda x:x.iloc[::-1].rolling(window=2, min_periods=1).sum().iloc[::-1])
    
        mlb_stats[col] = np.where(mlb_stats.PA<minimum_pa,
            mlb_stats[f'rolling_PA_x_{col}'] / mlb_stats['rolling_PA'],
            mlb_stats[col])
    
    mlb_stats['max_PA'] = mlb_stats.groupby('IDfg').rolling_PA.transform('max')
    for col in target_cols:
        mlb_stats[col] = np.where(mlb_stats.rolling_PA > minimum_pa, mlb_stats[col], np.nan)
            

    mlb_stats = mlb_stats[['IDfg', 'Season', 'max_PA'] + target_cols].copy()
    mlb_stats = mlb_stats.rename(columns={col: f'promotion_{col}' for col in target_cols})
        
    
    promotion_cols = [f'promotion_{col}' for col in target_cols]
    promotion_data = promotion_data.merge(mlb_stats.drop(columns='max_PA'), how='left', on=['IDfg', 'Season'])
    promotion_data = promotion_data.merge(mlb_stats[['IDfg','max_PA']].drop_duplicates(), how='left', on=['IDfg'])
    
    
    
    
    # --- Step 5: Handle next-season promotions
    next_season = promotion_data[promotion_data[[f'promotion_{col}' for col in target_cols]].isna().any(axis=1)].drop(columns=promotion_cols)
    
    
    
    # --- Step 6: Merge next-season MLB stats
    next_mlb_stats = mlb_stats.copy(deep=True)
    next_mlb_stats['Season'] -= 1
    
    next_season = next_season.merge(next_mlb_stats[['IDfg','Season']+promotion_cols], how='left', on=['IDfg', 'Season'])
    
    
    needs_next_season = promotion_data[promotion_data.promotion_wRC_plus.isna()][['IDfg','Season']]
    doesnt_need_next_season = promotion_data[~promotion_data.promotion_wRC_plus.isna()][['IDfg','Season']]
    next_season = next_season.merge(needs_next_season, how = 'inner', on =['IDfg','Season'])
    
    
    
    promotion_data = promotion_data.merge(doesnt_need_next_season, how = 'inner', on =['IDfg','Season'])
    
    # --- Step 9: Combine all usable promotion data
    promotion_data = pd.concat([promotion_data, next_season])
    
    
    promotion_data['max_PA'] = promotion_data['max_PA'].fillna(0)
    promotion_data['never_made_majors'] = np.logical_and(promotion_data.Season<2022,promotion_data.max_PA<400)
    
    # Calculate 20th percentile for each promotion column
    quantile_vals = promotion_data[promotion_cols].quantile(0.2)
    
    # Apply condition and fill
    for col in promotion_cols:
        q_val = quantile_vals[col]
        promotion_data.loc[promotion_data['never_made_majors'], col] = q_val



    promotion_data = promotion_data[['Season','Name','IDfg','Level','Age','never_made_majors', 'max_PA']\
        +[x+'_AAA' for x in feature_cols] +[x+'_AA' for x in feature_cols] + ['promotion_'+x for x in target_cols]]
    
    
    def wm(df, col):
        return (df[col+'_AAA'].fillna(0) * df.PA_AAA.fillna(0) + df[col+'_AA'].fillna(0) * df.PA_AA.fillna(0))/(df.PA_AAA.fillna(0)+df.PA_AA.fillna(0))
        
    
    promotion_data['HR_rate'] = wm(promotion_data, col='HR_rate')
    promotion_data['AVG'] = wm(promotion_data, col='AVG')
    promotion_data['OBP'] = wm(promotion_data, col='OBP')
    promotion_data['SLG'] = wm(promotion_data, col='SLG')
    promotion_data['PA'] = promotion_data.PA_AA.fillna(0)+promotion_data.PA_AAA.fillna(0)
    promotion_data['wRC_plus'] = wm(promotion_data, col='wRC_plus')
    
    
    
    prospect_report = pd.read_csv('prospect_report_data.csv').drop(columns = 'Unnamed: 0')
    promotion_data = promotion_data.merge(prospect_report, how = 'left', on = ['Name','Season'])
    promotion_data['wRC_plus'] = \
        (promotion_data['wRC_plus_AAA'].fillna(0) * promotion_data['PA_AAA'].fillna(0) + \
         .85*promotion_data['wRC_plus_AA'].fillna(0) * promotion_data['PA_AA'].fillna(0) )/promotion_data['PA']
    
    
    promotion_data['remaining_PA'] = np.sqrt((600-promotion_data.PA).clip(0,600))
    promotion_data['taken_PA'] = np.sqrt((promotion_data.PA).clip(0,600))
    
    promotion_data['wRC_plus_regressed'] = (promotion_data.remaining_PA*100 + promotion_data['wRC_plus']*promotion_data.taken_PA)/(promotion_data.remaining_PA + promotion_data.taken_PA)

    promotion_data['prospect_rank'] = np.where(promotion_data.Season>=2018, promotion_data.prospect_rank.fillna(-1), promotion_data.prospect_rank)

    final_feature_cols = ['Level','Age','wRC_plus_regressed']+[x+'_AAA' for x in feature_cols] +[x+'_AA' for x in feature_cols] + \
        [x for x in prospect_report.columns if x not in ['Name','Season']]
        
    if fit_model:
    
    
        train_data = promotion_data.dropna(subset=[f'promotion_{col}' for col in target_cols])
        
        train_data = train_data[train_data.Season <= 2021]

        train_data = train_data.fillna(0)
        
    
        X = pd.get_dummies(train_data[final_feature_cols], drop_first=True)
        Y = train_data[[f'promotion_{col}' for col in target_cols]]
        
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                verbosity=1,         # Optional: control output
                n_jobs=-1,           # Use all cores
                random_state=42,
                enable_categorical=True
            )
        )
        
        model.fit(X, Y)
        
        if overwrite:
            with open(model_path, 'wb') as f:
                pickle.dump(model,f)
                
    else:
        with open(model_path, 'rb') as f:
            model=pickle.load(f)
            
    data_for_pred = pd.get_dummies(promotion_data[final_feature_cols], drop_first=True).reindex(columns=model.feature_names_in_, fill_value=0).fillna(0)

    predictions = model.predict(data_for_pred)
    
    
    for i,col in enumerate(promotion_cols):
        promotion_data[col.replace('promotion','pred')] = predictions[:,i]
    
    promotion_data['pred_HR'] = promotion_data['pred_HR_rate']*promotion_data.PA
    promotion_data['pred_2B'] = promotion_data['pred_2B_rate']*promotion_data.PA
    promotion_data['pred_BB'] = promotion_data.PA*(promotion_data['pred_OBP'] - promotion_data['pred_hit_rate'])
    promotion_data['pred_BB%'] = promotion_data.pred_BB/promotion_data.PA
    
    promotion_data['pred_AVG'] =promotion_data.pred_hit_rate*promotion_data.PA/(promotion_data.PA - promotion_data.pred_BB)
    promotion_data['AB'] = promotion_data.PA - promotion_data.pred_BB

    return promotion_data


def get_rolling_columns(data, span=4):

    cols_to_keep = ['2B_rate','3B_rate','1B_rate', 'hit_rate','OBP', 'PA']
    
    historic_columns = ['BB%+','K%+','AVG+','OBP+','SLG+','ISO+','BB/K','HR_rate',
                        'wOBA','wRC_plus','Age', 'height','weight'] + cols_to_keep
    
    
    
    modern_columns = ['xBA', 'xSLG', 'xwOBA', 'wRC_plus', 'HR_rate','ISO+',
                      'Barrel%', 'Contact%', 'SwStr%', 'BB/K', 'maxEV', 'EV', 'LA', 'Age', 'height','weight'] + cols_to_keep

    columns = list(set(modern_columns + historic_columns))
    
    full_columns = columns.copy()

    for col in columns:

        if col not in ['height','weight']:
            data['PA_for_metric'] = np.where(data[col].isna(), np.nan, data['PA'])
    
            
            data['rolled_PA_for_metric'] = (
                data.sort_values('Season')
              .groupby('IDfg')['PA_for_metric']
              .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
            
            data[f'{col}_x_PA'] = data[col] * data.PA
            data[f'rolled_{col}'] = (
                data.sort_values('Season')
                  .groupby('IDfg')[f'{col}_x_PA']
                  .transform(lambda x: x.ewm(span=span, adjust=False).mean()) /
                data['rolled_PA_for_metric']
            )
            
            # EWMA numerator (weighted stat)
            data[f'rolled_std_{col}']= (
                data.groupby('IDfg')[f'{col}_x_PA']
                  .transform(lambda x: x.ewm(span=span, adjust=False).std())
                / data['rolled_PA_for_metric']
            )

            full_columns.append(f'rolled_{col}')
            full_columns.append(f'rolled_std_{col}')

    return data.drop(columns=[x for x in data.columns if '_x_' in x]), full_columns

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

def find_neighbors_for_player(df, full_columns, target_name, target_season, target_level, era, n_neighbors=5, n_components=5, feature_importance=False):

    df = df.copy()

    if era == 'historic':

        # no statcast data if minor leaguer so only use historic columns
        max_season = 2015 if target_level=='MLB' else 2030
        
        df = df[np.logical_or(np.logical_and(df.Season==target_season, df.Name==target_name), 
                              df.Season <max_season)]
    else:
        if target_season < 2015:
            return df.iloc[:0], None, None, None
            
        df = df[np.logical_or(np.logical_and(df.Season==target_season,df.Name==target_name), 
                              df.Season >=2015)]
    
    # Define columns
    historic_columns = ['BB%+','K%+','AVG+','OBP+','SLG+','ISO+','BB/K','HR_rate','wOBA','wRC_plus','Age', 'height','weight']

    modern_columns = ['xBA', 'xSLG', 'xwOBA', 'wRC_plus', 'HR_rate','ISO+','BB%+','K%+',
                      'Barrel%', 'Contact%', 'SwStr%', 'BB/K', 'maxEV', 'EV', 'LA', 'Age', 'height','weight']


    columns = historic_columns if era == 'historic' else modern_columns
    full_columns_ = columns + [f'rolled_{x}' for x in columns] + [f'rolled_std_{x}' for x in columns]
    full_columns = [x for x in full_columns_ if x in full_columns]

    full_columns.append('wRC_plus_prev')

    full_columns = list(set(full_columns))


    df = df[df.Season>=1978].sort_values('Season')

    df_clean = df.dropna(subset=full_columns).copy()

    df_clean = df_clean.reset_index(drop=True)
    
    # Find target player row after filtering
    target_idx = df_clean[
        (df_clean['Name'] == target_name) & (df_clean['Season'] == target_season)
    ].index

    if len(target_idx) == 0:
        return df.sort_values('distance').head(50), None, None, None
    if len(target_idx) == 0:
        raise ValueError(f"Player '{target_name}' in season {target_season} not found after filtering.")
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[full_columns])

    if feature_importance is not None:

        # Weight the standardized features
        X = X * feature_importance  # shape: (n_samples, n_features)

    # PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(X)

    for i in range(n_components):
        df_clean[f'PC{i+1}'] = pca_components[:, i]

    # Get the vector of the target player
    target_vector = pca_components[target_idx[0]].reshape(1, -1)



    n_neighbors = min(n_neighbors, df_clean.shape[0]-1)
    # Fit NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(pca_components)

    distances, indices = nn.kneighbors(target_vector)

    # Return full rows of the neighbors (excluding the player themself)
    neighbor_indices = indices[0][1:]  # skip self
    neighbors_df = df_clean.iloc[neighbor_indices].copy()
    neighbors_df['DistanceToTarget'] = distances[0][1:]

    cols_to_keep = ['2B_rate','3B_rate','1B_rate', 'hit_rate','OBP', 'PA', 'PA_next']
    df_cols = ['Name','Season','IDfg','DistanceToTarget','wRC_plus_next'] + list(set(historic_columns+modern_columns+full_columns) )
    df_cols += cols_to_keep
    return neighbors_df[df_cols], df, pca, full_columns
    

def build_distribution_for_player(data, player, season, columns,beta=.3, alpha=.5,show_plots = True, modern_feature_importances=None, historic_feature_importances=None):

    print(f"Building Distribution for {player} {season}")
    
    data = data.sort_values('Season')
                        
    data['wRC_plus_next'] = data.groupby('Name').wRC_plus.shift(-1)


    cdf = find_cluster(data, player, season)
    player_row = data[(data.Name == player) & (data.Season == season)].iloc[0]
    level = player_row.Level
    
    d=pd.DataFrame(player_row).T[columns]
    na_columns = d.isna().sum()
    
    # Display columns that have NA values
    full_columns_curr = list(pd.DataFrame(na_columns[na_columns == 0]).T.columns)

    historic_neighbors = 50 if level =='MLB' else 100
    modern_neighbors = 50 if level=='MLB' else 0
    
    
    ndf_historic, cdf_h, pca_h, full_columns_h = find_neighbors_for_player(cdf, full_columns_curr, target_name=player, 
                                                                           target_season=season, target_level=level, era='historic', n_neighbors=historic_neighbors, 
                                                                           n_components=5,feature_importance=historic_feature_importances)

    if level == 'MLB':
        ndf_modern,cdf_m, pca_m, full_columns_m  = find_neighbors_for_player(cdf, full_columns_curr, target_name=player,
                                                                         target_season=season, target_level=level, era='modern', n_neighbors=modern_neighbors,
                                                                         n_components=5, feature_importance=modern_feature_importances)
    
        ndf = pd.concat([ndf_modern, ndf_historic])
        cdf = pd.concat([cdf_m, cdf_h]) if (cdf_m is not None and cdf_h is not None) else cdf
    else:
        cdf = cdf_h if cdf_h is not None else cdf
        ndf = ndf_historic
    
    cdf = cdf.dropna(subset='wRC_plus_next').drop_duplicates(['Name','Season'])
    ndf = ndf.dropna(subset='wRC_plus_next').drop_duplicates(['Name','Season'])
    x = np.array(cdf.wRC_plus_next - cdf.rolled_wRC_plus).astype(float)
    y = np.array(ndf.wRC_plus_next - ndf.rolled_wRC_plus).astype(float)


    # Fit the skew normal
    a_fit1, loc_fit1, scale_fit1 = skewnorm.fit(x)
    a_fit2, loc_fit2, scale_fit2 = skewnorm.fit(y)
    
    # Predict (i.e., get PDF or samples)
    ln = np.linspace(-500, 500, len(cdf))
    

    # TODO empirically find value for this blend, rn this means 70% neighbors, 30% larger pool of less close matches

    n1 = int(len(cdf)*beta)
    n2 = int(len(cdf)*(1-beta))
    samples1 = skewnorm.rvs(a_fit1, loc=loc_fit1, scale=scale_fit1, size=n1)
    samples2 = skewnorm.rvs(a_fit2, loc=loc_fit2, scale=scale_fit2, size=n2)
    
    
    
    a_fit, loc_fit, scale_fit = skewnorm.fit(np.concatenate([samples1, samples2]))

    aging_curve  = pd.read_csv('aging_curve.csv')
   
    delta = a_fit / np.sqrt(1 + a_fit**2)
    
    # Original mean
    mu = loc_fit + scale_fit * delta * np.sqrt(2 / np.pi)

    aging_curve_shift = aging_curve[aging_curve.Age==player_row.Age].delta.iloc[0]
    mu_new = alpha * mu + (1-alpha) * aging_curve_shift

    # Adjusted loc
    loc_fit_adjusted = mu_new - scale_fit * delta * np.sqrt(2 / np.pi)

    
    
    pdf = skewnorm.pdf(ln, a_fit, loc=loc_fit_adjusted, scale=scale_fit)
    
    samples = skewnorm.rvs(a_fit, loc=loc_fit_adjusted, scale=scale_fit, size=1000)

    if show_plots:
        plt.plot(ln, pdf, label="Fitted PDF")
        plt.hist(x, bins=20, density=True, alpha=0.3, label="Data histogram")
        plt.hist(y, bins=20, density=True, alpha=0.3, label="Data histogram")
        plt.legend()
        plt.xlim(-100,100)
        plt.show()
    
        plt.hist(samples + player_row.rolled_wRC_plus, density=True)
        plt.axvline(x=player_row.rolled_wRC_plus, color = 'yellow')
        plt.axvline(x=player_row.wRC_plus_next, color = 'red')
        plt.show()
    
    return samples + player_row.rolled_wRC_plus, cdf, ndf, (a_fit, loc_fit_adjusted, scale_fit)



def get_baseline_projections(data, similar_players_df, player, season, dist_params, n_samples = 1000):

    print(f"Generating Projections for {player} {season}")

    data['singles_rate_change'] = data['1B_rate_next'] - data.rolled_1B_rate
    data['triples_rate_change'] = data['3B_rate_next'] - data.rolled_3B_rate
    data['doubles_rate_change'] = data['2B_rate_next'] - data.rolled_2B_rate
    data['OBP_change'] = data['OBP_next'] - data.rolled_OBP
    
    data['hit_rate_change'] = data['hit_rate_next'] - data.rolled_hit_rate
    data['HR_rate_change'] = data['HR_rate_next'] - data.rolled_HR_rate
    data['wRC_plus_change'] = data['wRC_plus_next'] - data.rolled_wRC_plus

    # === Helper to create inverse CDF ===
    def make_inv_cdf(real_deltas):
        sorted_vals = np.sort(real_deltas)
        ecdf = np.linspace(0, 1, len(sorted_vals))
        return interp1d(ecdf, sorted_vals, bounds_error=False,
                        fill_value=(sorted_vals[0], sorted_vals[-1]))
    
    # make these sttart from diff from rolled not diff from last yr
    data = data.sort_values('Season')
    
    data['HR_rate_next'] = data.groupby(['Name']).HR_rate.transform(lambda x:x.shift(-1))
    data['2B_rate_next'] = data.groupby(['Name'])['2B_rate'].transform(lambda x:x.shift(-1))
    data['OBP_next'] = data.groupby(['Name'])['OBP'].transform(lambda x:x.shift(-1))
    data['hit_rate_next'] = data.groupby(['Name'])['hit_rate'].transform(lambda x:x.shift(-1))
    
    
    data['doubles_rate_change'] = data['2B_rate_next'] - data.rolled_2B_rate
    data['HR_rate_change'] = data['HR_rate_next'] - data.rolled_HR_rate
    data['wRC_plus_change'] = data.wRC_plus_next - data.rolled_wRC_plus
    
    # change to rolled
    data['OBP_change'] = data['OBP_next'] - data.rolled_OBP
    data['hit_rate_change'] = data['hit_rate_next'] - data.rolled_hit_rate
    
    rate_df = similar_players_df[['Name','Season','PA', 'wRC_plus']].merge(data[['Name','Season', 'HR_rate_change',
                                                    'wRC_plus_change','doubles_rate_change','OBP_change','hit_rate_change']], 
                                                            how='left', on=['Name','Season'])
    
    rate_df = rate_df.dropna()
    
    # === Inputs ===
    a_fit, loc_fit, scale_fit = dist_params  # Skew-normal params for ΔwRC+
    
    # Empirical diffs
    
    
    # Construct correlation matrix for ΔwRC+, ΔAVG, ΔBB%, ΔHR%, Δ2B%
    corr_matrix = rate_df[['wRC_plus_change', 'hit_rate_change', 'OBP_change',
                           'HR_rate_change', 'doubles_rate_change']].corr().values
    
    
    corr_PA = rate_df[['wRC_plus_change', 'PA']].corr().iloc[0,1]
    
    
    
    rate_df = rate_df.merge(similar_players_df[['Name','Season']], how = 'inner')
    
    real_delta_wrc = rate_df['wRC_plus_change']
    real_delta_avg = rate_df['hit_rate_change']
    real_delta_obp = rate_df['OBP_change']
    real_delta_hr = rate_df['HR_rate_change']
    real_delta_2b = rate_df['doubles_rate_change']
    real_pa = rate_df['PA']
    
    
    
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.randn(n_samples, 5)
    
    ''' Start PA finder'''

    
    # note this is done on corr to change in wRC+ but should be changed to wRC+ after delta
    z_pa = corr_PA * z[:, 0] + np.sqrt(1 - corr_PA**2) * np.random.randn(n_samples)
    
    sorted_pa = np.sort(real_pa)
    ranks = np.linspace(0, 1, len(sorted_pa))
    inverse_cdf_pa = interp1d(ranks, sorted_pa, bounds_error=False, fill_value=(sorted_pa[0], sorted_pa[-1]))
    
    u2 = norm.cdf(z_pa)
    PAs = inverse_cdf_pa(u2)
    
    ''' End PA finder'''
    
    z_corr = z @ L.T  # Correlated normals
    
    # === Step 2: Transform to ΔwRC+ using skew-normal ===
    u1 = norm.cdf(z_corr[:, 0])
    delta_wrc = skewnorm.ppf(u1, a=a_fit, loc=loc_fit, scale=scale_fit)


    
    # === Step 3: Transform ΔAVG and ΔBB% ===
    inverse_cdf_avg = make_inv_cdf(real_delta_avg)
    inverse_cdf_obp = make_inv_cdf(real_delta_obp)
    
    u2 = norm.cdf(z_corr[:, 1])
    u3 = norm.cdf(z_corr[:, 2])
    delta_avg = inverse_cdf_avg(u2)
    delta_obp = inverse_cdf_obp(u3)
    
    # === Step 4: HR% based on wRC+ and AVG ===
    z_hr = z_corr[:, 3]
    u4 = norm.cdf(z_hr)
    inverse_cdf_hr = make_inv_cdf(real_delta_hr)
    delta_hr = inverse_cdf_hr(u4)
    
    # === Step 5: 2B% based on wRC+, AVG, HR% ===
    z_2b = z_corr[:, 4]
    u5 = norm.cdf(z_2b)
    inverse_cdf_2b = make_inv_cdf(real_delta_2b)
    delta_2b = inverse_cdf_2b(u5)
    
    pdf = data[np.logical_and(data.Name == player, data.Season==season)]

    projection_dict = {}
    projection_dict['rolled_wRC_plus'] = float(pdf['rolled_wRC_plus'].iloc[0])
    projection_dict['wRC_plus'] = float(pdf['rolled_wRC_plus'].iloc[0]) + delta_wrc
    projection_dict['hit_rate'] = float(pdf['rolled_hit_rate'].iloc[0]) + delta_avg
    projection_dict['HR_rate']  = float(pdf['rolled_HR_rate'].iloc[0]) + delta_hr
    projection_dict['2B_rate'] = float(pdf['rolled_2B_rate'].iloc[0])  + delta_2b
    projection_dict['3B_rate'] = float(pdf['3B'].fillna(0).iloc[0]/pdf.PA.iloc[0]) + 0
    projection_dict['1B_rate'] = projection_dict['hit_rate'] - projection_dict['HR_rate'] - projection_dict['3B_rate'] - projection_dict['2B_rate']
    
    projection_dict['OBP'] = float(pdf['rolled_OBP'].iloc[0]) + delta_obp
    
    projection_dict['BB_rate'] = projection_dict['OBP'] - projection_dict['hit_rate']
    
    projection_dict['BB'] = projection_dict['BB_rate']*PAs
    projection_dict['HR'] = projection_dict['HR_rate']*PAs
    projection_dict['2B'] = projection_dict['2B_rate']*PAs
    projection_dict['3B'] = projection_dict['3B_rate']*PAs
    projection_dict['1B'] = projection_dict['1B_rate']*PAs
    projection_dict['H'] = projection_dict['hit_rate']*PAs
    
    projection_dict['PA'] = PAs
    projection_dict['AB'] = PAs - projection_dict['BB']
    
    projection_dict['AVG'] = projection_dict['H'] / projection_dict['AB']

    projection_dict['TB']= (4*projection_dict['HR'] + 3 * projection_dict['3B']+ 2 * projection_dict['2B']  + 1 * projection_dict['1B'])
    projection_dict['SLG'] = projection_dict['TB']/projection_dict['AB']
    
    projection_dict['OPS'] = projection_dict['OBP'] + projection_dict['SLG']
    projection_df = pd.DataFrame(projection_dict)
    
    projection_df['Name'] = player
    projection_df['Season'] = season + 1
    projection_df['Sample'] = range(1,1+len(projection_df))
    projection_df[['Name','Season', 'Sample'] + [x for x in projection_df if x not in ['Name','Season','Sample']]].round(3)
    return projection_df


if __name__ == '__main__':

    LOAD_FROM_CACHE = False
    OVERWRITE = True
    
    if LOAD_FROM_CACHE:
    
        with open('columns.pkl','rb') as f:
            columns = pickle.load(f)
        data = pd.read_csv('cached_data_obj.csv')
        
    else:
        data = prepare_data(path = '../data.csv')
        
        data, columns = get_rolling_columns(data)
    
        if OVERWRITE:
            
            with open('columns.pkl','wb') as f:
                pickle.dump(columns,f)
                
            data.to_csv('cached_data_obj.csv')

    smpl = data[np.logical_and(data.PA>50, data.Season==2024)]
    
    l=[]
    i=0
    for player, season in tqdm.tqdm(smpl[['Name','Season']].values):
    
 
        samples, cdf, ndf, dist_params = build_distribution_for_player(data, player=player, season=season, columns=columns, show_plots = False)

        projection_df = get_baseline_projections(data, cdf, player, season, dist_params, n_samples = 1000)
        l.append(projection_df)

        if i%10==0:
            pd.concat(l).to_csv('predictions.csv')
            
        i +=1
    pd.concat(l).to_csv('predictions.csv')
