import pandas as pd


def get_team_df(
    player_df):
    team_df = player_df.groupby(['matchid', 'team_name']).first().reset_index(drop = False)
    team_df['date'] = pd.to_datetime(team_df['date'])
    return team_df


def add_features_football(
    team_df):
    team_df = team_df.sort_values(by = ['date'])
    team_df['win'] = team_df['result'].map({'W': 1, 'D': 0, 'L': 0})
    team_df['draw'] = team_df['result'].map({'W': 0, 'D': 1, 'L': 0})
    team_df['loss'] = team_df['result'].map({'W': 0, 'D': 0, 'L': 1})
    team_df['nr_wins'] = \
        team_df.groupby(['team_name', 'season'])['win'].transform(lambda x: x.cumsum())
    team_df['game_nr'] = \
        team_df.groupby(['team_name', 'season']).cumcount() + 1
    team_df['win_pct'] = \
        team_df['nr_wins']/team_df['game_nr']
    team_df['goals_margin'] = team_df['goals'] - team_df['opp_goals']
    return team_df

def add_features_nba(
    team_df,
    group_vars):
    team_df['win'] = team_df['wl'].map({'W': 1, 'L': 0})
    team_df['nr_wins'] = \
        team_df.groupby(group_vars)['win'].transform(lambda x: x.cumsum())
    team_df['game_nr'] = \
        team_df.groupby(group_vars).cumcount() + 1
    team_df['win_pct'] = \
        team_df['nr_wins']/team_df['game_nr']
    return team_df


def add_lagged_features(
    team_df,
    feat_lag_dict,
    group_vars=['team_name', 'season']):
    for feat in feat_lag_dict.keys():
        for lag in feat_lag_dict[feat]:
            team_df[feat + '_lag_' + str(lag)] = \
                team_df.groupby(group_vars)[feat].transform(lambda x: x.shift(lag))
    return team_df


def get_match_df(team_df):
    team_df['homeoraway'] = team_df['home'].map({1: 'home', 0: 'away'})
    team_df.set_index(['matchid', 'homeoraway'], inplace = True, drop = False)
    match_df = team_df.unstack('homeoraway')
    match_df.columns = ['_'.join(col).strip() for col in match_df.columns.values]

    # These vars are the same for both teams
    match_vars = ["league", "season", "date"]
    for var in match_vars:
        match_df[var] = match_df[var + "_home"]
 
    # Add numerical outcome variable
    match_df['result_num'] = match_df['result_home'].map({'W': 2, 'D': 1, 'L': 0})

    return match_df


def get_match_df_nba(df):
    joined = pd.merge(df, df, suffixes=['_home', '_away'],
                      on=['season_id', 'game_id', 'game_date'])
    result = joined[joined.team_id_home != joined.team_id_away]
    result = result[result.matchup_home.str.contains(' vs. ')]

    return result


def get_bet_probs(
    avail_bet_data,
    odds_vars = ['B365H', 'B365D', 'B365A'],
    odds_path = "C:/Users/peter/data/betting_data",
    ):

    odds_df = pd.DataFrame()

    for league in avail_bet_data.keys():
        for season in avail_bet_data[league]:
            temp_df = pd.read_csv(odds_path + "/" + league + '_' + season + '.csv')
            temp_df["league"] = league
            temp_df["season"] = season
            temp_df["team_name_home"] = temp_df.HomeTeam.map(team_dict).fillna(temp_df['HomeTeam'])
            temp_df["team_name_away"] = temp_df.AwayTeam.map(team_dict).fillna(temp_df['AwayTeam'])
            
            odds_df = odds_df.append(temp_df[['league', 'season', 'team_name_home', 'team_name_away'] + odds_vars])
    
    odds_df.reset_index(inplace = True, drop = True)

    # Transform odds to probabilities
    odds_df[odds_vars] = 1 / odds_df[odds_vars]

    return odds_df


# Keys as in betting data, values as in scraped data
team_dict = {
'Man City': 'Manchester City', 
'Man United': 'Manchester United',
'West Brom': 'West Bromwich Albion',
'QPR': 'Queens Park Rangers',
'Wolves': 'Wolverhampton Wanderers',
'Newcastle': 'Newcastle United',
'Milan': 'AC Milan',
'Siena': 'Robur Siena',
'Dortmund': 'Borussia Dortmund',
'Hertha': 'Hertha Berlin',
'Mainz': 'Mainz 05',
'Nurnberg': 'Nuernberg',
'Ein Frankfurt': 'Eintracht Frankfurt',
'FC Koln': 'FC Cologne',
'Hamburg': 'Hamburger SV', 
'Hannover': 'Hannover 96',
'Leverkusen': 'Bayer Leverkusen',
'Stuttgart': 'VfB Stuttgart',
"M'gladbach": 'Borussia M.Gladbach',
'Greuther Furth': 'Greuther Fuerth',
'Fortuna Dusseldorf': 'Fortuna Duesseldorf',
'Braunschweig': 'Eintracht Braunschweig',
'RB Leipzig': 'RasenBallsport Leipzig',
'St Pauli': 'St. Pauli',
'Spal': 'SPAL 2013',
'Parma': 'Parma Calcio 1913'
}