import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
from utils_dataprep import get_team_df, add_features_football, add_lagged_features,\
    get_match_df, get_bet_probs
from utils_eval import get_feature_importance, get_act_pred

np.random.seed(123)

#%%
# Data load and feature engineering

# Load data from pickle
df = pickle.load(open("C:/Users/peter/data/player_level.pckl", "rb"))

# Filter variables
info_vars = ['home', 'league', 'season', 'date', 'matchid', 'team_name']
feat_vars = ['goals', 'opp_goals']
outcome = ['result']

# key of player_df: p_name & matchid
player_df = df[info_vars + feat_vars + outcome]

# Create team level dataset
team_df = get_team_df(player_df)
team_df = add_features_football(team_df)
feat_lag_dict = {
    'win_pct': [1],
    'goals': [1,2,3],
    'opp_goals': [1,2,3],
    'win': [1,2,3]
}
# key of team_df: team_name & matchid
team_df = add_lagged_features(team_df, feat_lag_dict)

# aggregate to match_df with key: matchid
match_df = get_match_df(team_df)

# Create categorical variables
match_df['league_cat'] = match_df['league'].astype('category').cat.codes
match_df['team_name_home_cat'] = match_df['team_name_home'].astype('category').cat.codes
match_df['team_name_away_cat'] = match_df['team_name_away'].astype('category').cat.codes


#%%
# Add betting data

# Specify leagues and seasons for which odds data is available
avail_bet_data = {
    'prem_league':  ['09_10', '10_11', '11_12', '12_13', '13_14', '14_15', '15_16', '16_17', '17_18', '18_19'],
    'bundesliga':   ['09_10', '10_11', '11_12', '12_13', '13_14', '14_15', '15_16', '16_17', '17_18', '18_19'],
    'serie_a':      ['09_10', '10_11', '11_12', '12_13', '13_14', '14_15', '15_16', '16_17', '17_18', '18_19']
}

odds_vars = ['B365H', 'B365D', 'B365A']

# Retrieve betting data
odds_df = get_bet_probs(
    avail_bet_data = avail_bet_data,
    odds_vars = odds_vars
    )

# Merge betting data to match data
match_df = match_df.merge(
    odds_df,
    how = 'left',
    on = ['league', 'season', 'team_name_home', 'team_name_away']
)


#%%
# Filter and split data

# Use only games after burn_in_period
burn_in_period = 5
match_df = match_df.loc[(match_df.game_nr_away > burn_in_period) & (match_df.game_nr_home > burn_in_period)]

# Column conditions
lagged_features = [feat for feat in match_df.columns if '_lag_' in feat]
categoricals = ['league_cat', 'team_name_home_cat', 'team_name_away_cat']
cols = lagged_features + categoricals + odds_vars
label = 'result_num'

row_mask = ~match_df[cols + [label]].isnull().any(axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    match_df[row_mask][cols], 
    match_df[row_mask]['result_num'], 
    test_size=0.1)


#%%
lgb_train = lgb.Dataset(X_train, y_train,
                       free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_classes': 3,
    # 'num_leaves': 31,
    # 'learning_rate': 0.05,
    # 'feature_fraction': 0.9,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                #valid_sets=lgb_train,  # eval training data
                feature_name=match_df[row_mask][cols].columns.to_list(),
                categorical_feature=[2]
                )


#%%
# Predict on test sample
preds_df = pd.DataFrame(gbm.predict(X_test), columns = ["pred_prob_L", "pred_prob_D", "pred_prob_W"], index = X_test.index)
preds_df['pred'] = preds_df.idxmax(axis=1)

merged_df = match_df.join(preds_df, how='inner')

get_feature_importance(gbm)
get_act_pred(preds_df)

# AUC
test_auc = roc_auc_score(
    y_true=preds_df['act'],
    y_score=preds_df[['L', 'D', 'W']],
    multi_class='ovo'
)
print("Test AUC:" + str(test_auc))

# Does it make money?

def calc_return(pred_probs, bet_probs, true_results, betting_cutoff):

    diff_probs = np.array(pred_probs) - np.array(bet_probs)
    betting_mask = diff_probs > betting_cutoff

    payouts_all = betting_mask*np.array(1/bet_probs)
    
    # For which game was a bet placed and the prediction was true?
    hit_idx = np.sum(betting_mask > 0, axis = 1).astype(bool) & (np.argmax(betting_mask, axis = 1) == true_results)

    nr_bets = betting_mask.sum()    

    if nr_bets > 0:
        revenue = np.sum(payouts_all[hit_idx])
        roi = (revenue - nr_bets)/nr_bets
        
        return roi, nr_bets
    else:
        return np.nan, np.nan


# Plot ROI
roi = []
perc_bet = []

cutoff_range = np.arange(0,0.3,0.005)

for cutoff in cutoff_range:
    temp_roi, temp_perc_bet = calc_return(
        pred_probs=merged_df[['pred_prob_L', 'pred_prob_D', 'pred_prob_W']],
        bet_probs=merged_df[['B365A', 'B365D', 'B365H']],
        true_results=merged_df['result_home'].map({'L':0, 'D':1, 'W':2}),
        betting_cutoff=cutoff
    )
    roi.append(temp_roi)
    perc_bet.append(temp_perc_bet)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Betting Cutoff')
ax1.set_ylabel('ROI in %', color=color)
ax1.plot(cutoff_range, roi, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(cutoff_range, [0]*len(cutoff_range), color = color, linestyle = '--')

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('% of Games with Bet', color=color)
ax2.plot(cutoff_range, perc_bet, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
