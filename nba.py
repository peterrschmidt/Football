import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
from nba_api.stats.endpoints import leaguegamefinder
from utils_dataprep import add_lagged_features, add_features_nba, get_match_df_nba
from utils_eval import get_feature_importance, get_act_pred


#%%
# Pull data
seasons = ["2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19"]

team_df = pd.DataFrame()

for season in seasons:
    finder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable="Regular Season"
    )

    team_df = team_df.append(finder.get_data_frames()[0])

nba_group_vars = ['season_id', 'team_id']

#%% Preprocessing
team_df.columns = map(str.lower, team_df.columns)
team_df['game_date'] = pd.to_datetime(team_df['game_date'])

# Order by date
team_df = team_df.sort_values(by='game_date')

#%% Feature Engineering
# Add features on team level
team_df = add_features_nba(team_df, group_vars=nba_group_vars)

# Add lagged features
nba_feat_lag_dict = {
    "win_pct": [1],
    'win': [1,2,3,4,5],
    'pts': [1,2,3,4,5],
    'reb': [1,2,3,4,5],
    'ast': [1,2,3,4,5],
    'stl': [1,2,3,4,5],
    'blk': [1,2,3,4,5],
    'tov': [1,2,3,4,5],
    'pf': [1,2,3,4,5],
    "plus_minus": [1,2,3,4,5]
}

team_df = add_lagged_features(
    team_df,
    feat_lag_dict=nba_feat_lag_dict,
    group_vars=nba_group_vars)

# Transform to match df
match_df = get_match_df_nba(team_df)
match_df.set_index('game_id', inplace = True, drop = True)

# Add team categorical variables
match_df['team_name_home_cat'] = match_df['team_abbreviation_home'].astype('category').cat.codes
match_df['team_name_away_cat'] = match_df['team_abbreviation_away'].astype('category').cat.codes

#%% Filter data

# Column conditions
lagged_features = [feat for feat in match_df.columns if '_lag_' in feat]
categoricals = ['team_name_home_cat', 'team_name_away_cat']
label = 'win_home'
features = categoricals + lagged_features

# Use only games after burn_in_period and filter missings
burn_in_period = 10
row_filter = (
    ((match_df.game_nr_away > burn_in_period) & (match_df.game_nr_home > burn_in_period)) &
    ~match_df[features + [label]].isnull().any(axis=1)
)

df_est = match_df[row_filter][features + [label]]

#%% Split data
# TT Split
#np.random.seed(234)
test_size = 0.2
test_mask = np.random.binomial(1, test_size, df_est.shape[0]).astype(bool)

X_train = df_est.loc[~test_mask, features]
X_test = df_est.loc[test_mask, features]
y_train = df_est.loc[~test_mask, label]
y_test = df_est.loc[test_mask, label]

lgb_train = lgb.Dataset(X_train, y_train)


#%% Parameter tuning
# ht_df = pd.DataFrame()

# lr_cands = [0.01, 0.05, 0.08, 0.1, 0.15, 0.2]
# nl_cands = np.arange(2,10)
# md_cands = np.arange(4,8)
# nbr_cands = [100, 500, 1000]

# for lr_cand in lr_cands:
#     for nl_cand in nl_cands:
#         for md_cand in md_cands:
#             for nbr_cand in nbr_cands:
            
#                 print(
#                     'learning_rate: ' + str(lr_cand) + "\n" + 
#                     'num_leaves: ' + str(nl_cand) + "\n" + 
#                     'max_depth: ' + str(md_cand) + "\n" + 
#                     'num_boost_round: ' + str(nbr_cand)
#                 )

#                 params = {
#                     'objective': 'binary',
#                     'learning_rate': lr_cand,
#                     'num_leaves': nl_cand,
#                     'max_depth': md_cand
#                 }

#                 gbm = lgb.train(params,
#                                 lgb_train,
#                                 num_boost_round=nbr_cand,
#                                 feature_name=X_train.columns.to_list(),
#                                 categorical_feature=(0,1)
#                                 )
                
#                 preds = pd.DataFrame({
#                     "prob_home_win": gbm.predict(df_est[features]),
#                     "test": test_mask
#                     }, index = df_est.index)

#                 joined = pd.DataFrame(match_df[row_filter].loc[test_mask, 'wl_home']).join(preds, how="inner")

#                 test_auc = roc_auc_score(
#                     y_true=joined['wl_home'],
#                     y_score=joined["prob_home_win"]
#                 ) 


#                 ht_df = ht_df.append({
#                     'learning_rate': lr_cand,
#                     'num_leaves': nl_cand,
#                     'num_boost_round': nbr_cand,
#                     'max_depth': md_cand,
#                     'test_auc': test_auc
#                 }, ignore_index=True)


# ht_df.loc[ht_df.test_auc == ht_df.test_auc.max()]

#%% Train model
params = {
    'objective': 'binary',
    'learning_rate': 0.01,
    'num_leaves': 2,
    #'max_depth':
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                feature_name=X_train.columns.to_list()
                )      

#%% Actual vs Predicted
# Add predictions
preds = pd.DataFrame({
    "prob_home_win": gbm.predict(df_est[features]),
    "test": test_mask
    }, index = df_est.index)

match_df = match_df.join(preds, how="left")

train_tp = sum((match_df.loc[match_df['test'] == False, 'prob_home_win'] >= 0.5) & (match_df.loc[match_df['test'] == False, 'wl_home'] == "W"))
train_tn = sum((match_df.loc[match_df['test'] == False, 'prob_home_win'] < 0.5) & (match_df.loc[match_df['test'] == False, 'wl_home'] == "L"))

test_tp  = sum((match_df.loc[match_df['test'] == True, 'prob_home_win'] >= 0.5) & (match_df.loc[match_df['test'] == True, 'wl_home'] == "W"))
test_tn  = sum((match_df.loc[match_df['test'] == True, 'prob_home_win'] < 0.5) & (match_df.loc[match_df['test'] == True, 'wl_home'] == "L"))

n_train = match_df.loc[match_df['test'] == False].shape[0]
n_test = match_df.loc[match_df['test'] == True].shape[0]

print("% Train correctly predicted: " + 
    str(round((train_tn + train_tp)/n_train, 4))
)

print("% Test correctly predicted: " + 
    str(round((test_tp + test_tn)/n_test, 4))
)

#%% Diagnostics
get_feature_importance(gbm)

# AUC
train_auc = roc_auc_score(
    y_true=match_df.loc[match_df['test'] == False, 'win_home'],
    y_score=match_df.loc[match_df['test'] == False, 'prob_home_win']
)

print("Train AUC:" + str(train_auc))

test_auc = roc_auc_score(
    y_true=match_df.loc[match_df['test'] == True, 'win_home'],
    y_score=match_df.loc[match_df['test'] == True, 'prob_home_win']
)

print("Test AUC:" + str(test_auc))
 

#%%
odds_df = pd.read_csv("C:/Users/peter/data/nba/nba_betting_money_line.csv")
odds_df = odds_df.loc[odds_df.book_name == "5Dimes", ['game_id', 'price1', 'price2']]

match_df['game_id'] = match_df.index.astype(int)

match_df.reset_index(drop=True, inplace=True)

merged = pd.merge(match_df, odds_df, how='inner', on="game_id")

#%%

def conv_odds(row):
    absrow = abs(row)
    if row >= 0:
        odds = (absrow / 100) + 1
    else:
        odds = (100 / absrow) + 1
    return 1/odds

merged['odds_home_win'] = merged['price1'].apply(conv_odds)
merged['odds_home_loss'] = merged['price2'].apply(conv_odds)


#%% Does it make money?

def calc_return(pred_probs, bet_probs, true_results, betting_cutoff):

    diff_probs = abs(np.array(pred_probs) - np.array(bet_probs))
    betting_mask = diff_probs > betting_cutoff

    payouts_all = betting_mask*np.array(1/bet_probs)
    
    # For which game was a bet placed and the prediction was true?
    hit_idx = betting_mask * (((pred_probs >= 0.5) & (true_results == "W")) | ((pred_probs < 0.5) & (true_results == "L")))

    nr_bets = betting_mask.sum()

    if nr_bets > 0:
        revenue = np.sum(payouts_all[hit_idx])
        roi = (revenue - nr_bets)/nr_bets
        
        return roi, nr_bets
    else:
        return np.nan, np.nan

pred_probs = merged.loc[merged.test == True, "prob_home_win"]
bet_probs = merged.loc[merged.test == True, "odds_home_win"]
true_results = merged.loc[merged.test == True, "wl_home"]

calc_return(
    pred_probs,
    bet_probs,
    true_results,
    betting_cutoff=0.1
)