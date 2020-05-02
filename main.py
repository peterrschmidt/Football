import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils_dataprep import get_team_df, add_features, add_lagged_features, get_match_df,\
    add_bet_probs

# Load data from pickle
df = pickle.load(open("C:/Users/peter/data/player_level.pckl", "rb"))

# Filter variables
info_vars = ['home', 'league', 'season', 'date', 'matchid', 'team_name']
feat_vars = ['goals', 'opp_goals']
outcome = ['result']

# key of player_df: p_name & matchid
player_df = df[info_vars + feat_vars + outcome]

# Create team level dataset and add features
team_df = get_team_df(player_df)
team_df = add_features(team_df)
feat_lag_dict = {
    'win_pct': [1],
    # 'goals': [1,2],
    # 'opp_goals': [1,2]
}
# key of team_df: team_name & matchid
team_df = add_lagged_features(team_df, feat_lag_dict)

# key of match_df: matchid
match_df = get_match_df(team_df)

# Add columns with prob of Home Win, Away Win and Draw
match_df = add_bet_probs(match_df)

match_df.loc[(pd.isna(match_df['B365H'])) & (match_df.season == "15_16"), "league"].unique()
match_df.loc[np.argmax(match_df['B365H'])]

# Create df with features and series with labels
col_mask = [feat for feat in match_df.columns if '_lag_' in feat]
row_mask = ~match_df[col_mask].isnull().any(axis=1)

features = match_df[row_mask][col_mask]
labels = match_df[row_mask]['win_home']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.33, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train,
                       free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                feature_name=features.columns.to_list()
                )

print('Feature names:', gbm.feature_name())
print('Feature importances:', list(gbm.feature_importance()))

y_pred = gbm.predict(X_test)
print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

TP = y_test[(y_pred > 0.5) & (y_test == 1)].shape[0]
FP = y_test[(y_pred > 0.5) & (y_test == 0)].shape[0]
TN = y_test[(y_pred < 0.5) & (y_test == 0)].shape[0]
FN = y_test[(y_pred < 0.5) & (y_test == 1)].shape[0]

print("TP: " + str(TP) + " FP: " + str(FP) + "\n"
      "FN: " + str(FN) + " TN: " + str(TN)
      )

print("Perc. corr. predicted: " + str(round((TP+TN)/y_test.shape[0], 4)))

team_df.loc[y_test.index[np.argmax(y_pred)]]

team_df.loc[y_test.index[np.argmin(y_pred)]]