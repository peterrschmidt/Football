import pandas as pd


def get_team_df(
    player_df):
    team_df = player_df.groupby(['matchid', 'team_name']).first().reset_index(drop = False)
    team_df['date'] = pd.to_datetime(team_df['date'])
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


def get_match_df(team_df):
    team_df['homeoraway'] = team_df['home'].map({1: 'home', 0: 'away'})
    team_df.set_index(['matchid', 'homeoraway'], inplace = True, drop = False)
    match_df = team_df.unstack('homeoraway')
    match_df.columns = ['_'.join(col).strip() for col in match_df.columns.values]

    # These vars are the same for both teams
    match_vars = ["league", "season", "date"]
    for var in match_vars:
        match_df[var] = match_df[var + "_home"]

    # These vars can be dropped
    for drop_var in match_vars + ["matchid", "home", "homeoraway"]:
        match_df.drop(drop_var + "_home", axis = 1, inplace = True)
        match_df.drop(drop_var + "_away", axis = 1, inplace = True)
        
    return match_df


def add_bet_probs(
    match_df,
    odds_path = "C:/Users/peter/data/all-euro-data-2015-2016.xls",
    merge_vars = ['league', 'season', 'team_name_home', 'team_name_away'],
    odds_vars = ['B365H', 'B365D', 'B365A']
):

    odds_data = pd.ExcelFile(odds_path)
    odds_df = pd.DataFrame([])
    for league in match_df.league.unique():
        if league in league_dict.keys():
            temp_df = odds_data.parse(sheet_name = league_dict[league])
            temp_df["league"] = league
            temp_df["season"] = "15_16" #TODO: make flexible
            temp_df["team_name_home"] = temp_df.HomeTeam.map(team_dict).fillna(temp_df['HomeTeam'])
            temp_df["team_name_away"] = temp_df.AwayTeam.map(team_dict).fillna(temp_df['AwayTeam'])
        
        odds_df = pd.concat([odds_df, temp_df], ignore_index = True)
    
    # Transform odds to probabilities
    for var in odds_vars:
        odds_df[var] = 1/odds_df[var]

    match_df = match_df.merge(
        odds_df[merge_vars + odds_vars],
        how = 'left',
        on = merge_vars
    )

    return match_df



# Merge with odds data
league_dict = {
    "prem_league": "E0",
    "bundesliga": "D1",
    "serie_a": "I1",
    # "laliga": "SP1",
    # "ligue1": "F1",
    # "superlig": "T1"
}

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
'RB Leipzig': 'RasenBallsport Leipzig'
}