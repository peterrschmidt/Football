import pandas as pd
import numpy as np
import pickle
from utils_dataprep import get_team_df, add_features, add_lagged_features, get_feats_labels

# Load data from pickle
df = pickle.load(open("C:/Users/peter/data/player_level.pckl", "rb"))

# Filter variables
info_vars = ['home', 'league', 'season', 'date', 'matchid', 'team_name']
feat_vars = ['goals', 'opp_goals']
outcome = ['result']

player_df = df[info_vars + feat_vars + outcome]

# Create team level dataset and add features
team_df = get_team_df(player_df)
team_df = add_features(team_df)
feat_lag_dict = {
    'win_pct': [1],
    'goals': [1,2],
    'opp_goals': [1,2]
}
team_df = add_lagged_features(team_df, feat_lag_dict)

# team_df.loc[(team_df.team_name == "Bayern Munich") & (team_df.season == "15_16")]

# Create df with features and series with labels
features, labels = get_feats_labels(team_df)
