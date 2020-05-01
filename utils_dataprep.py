import pandas as pd


def get_team_df(
    player_df):
    team_df = player_df.groupby(['matchid', 'team_name']).first().reset_index(drop = False)
    team_df['date'] = pd.to_datetime(team_df['date'])
    team_df['homeoraway'] = team_df['home'].map({1: 'home', 0: 'away'})
    return team_df


def add_features(
    team_df):
    team_df = team_df.sort_values(by = ['date'])
    team_df['win'] = team_df['result'].map({'W': 1, 'D': 0, 'L': 0})
    team_df['nr_wins'] = \
        team_df.groupby(['team_name', 'season'])['win'].transform(lambda x: x.cumsum())
    team_df['game_nr'] = \
        team_df.groupby(['team_name', 'season']).cumcount() + 1
    team_df['win_pct'] = \
        team_df['nr_wins']/team_df['game_nr']
    team_df['goals_margin'] = team_df['goals'] - team_df['opp_goals']
    return team_df


def add_lagged_features(
    team_df,
    feat_lag_dict):
    for feat in feat_lag_dict.keys():
        for lag in feat_lag_dict[feat]:
            team_df[feat + '_lag_' + str(lag)] = \
                team_df.groupby(['team_name', 'season'])[feat].transform(lambda x: x.shift(lag))
    return team_df


def get_feats_labels(
team_df
):
    team_df.set_index(['matchid', 'homeoraway'], inplace = True, drop = False)
    match_df = team_df.unstack('homeoraway')
    match_df.columns = ['_'.join(col).strip() for col in match_df.columns.values]

    col_mask = [feat for feat in match_df.columns if '_lag_' in feat]
    row_mask = ~ match_df[col_mask].isnull().any(axis=1)

    features = match_df[row_mask][col_mask]
    labels = match_df[row_mask]['win_home']

    return features, labels
    