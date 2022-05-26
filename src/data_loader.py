import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"

EVENTMSGTYPE_dict = {
    1: "FIELD_GOAL_MADE",
    2: "FIELD_GOAL_MISSED",
    3: "FREE_THROW",
    4: "REBOUND",
    5: "TURNOVER",
    6: "FOUL",
    7: "VIOLATION",
    8: "SUBSTITUTION",
    9: "TIMEOUT",
    10: "JUMP_BALL",
    11: "EJECTION",
    12: "PERIOD_BEGIN",
    13: "PERIOD_END"
}

PERSONTYPE_dict = {
    0: 0,
    1: "TIMEOUT",
    2: "HOME_TEAM", #If the play is not attributed to a single player
    3: "VISITOR_TEAM", #If the play is not attributed to a single player
    4: "HOME_PLAYER", #If the play is not attributed to a single player
    5: "VISITOR_PLAYER", #If the play is not attributed to a single player
    6: "HOME_TEAM_FOUL",
    7: "VISITOR_TEAM_FOUL"
}

EVENTMSGACTIONTYPE_FIELD_GOAL_dict = {
    0: "No Shot",
    1: "Jump Shot",
    2: "Running Jump Shot",
    3: "Hook Shot",
    4: "Tip Shot",
    5: "Layup",
    6: "Driving Layup",
    7: "Dunk",
    8: "Slam Dunk",
    9: "Driving Dunk",
    40: "Layup",
    41: "Running Layup",
    42: "Driving Layup",
    43: "Alley Oop Layup",
    44: "Reverse Layup",
    45: "Jump Shot",
    46: "Running Jump Shot",
    47: "Turnaround Jump Shot",
    48: "Dunk",
    49: "Driving Dunk",
    50: "Running Dunk",
    51: "Reverse Dunk",
    52: "Alley Oop Dunk",
    53: "Tip Shot",
    54: "Running Tip Shot",
    55: "Hook Shot",
    56: "Running Hook Shot",
    57: "Driving Hook Shot",
    58: "Turnaround Hook Shot",
    59: "Finger Roll",
    60: "Running Finger Roll",
    61: "Driving Finger Roll",
    62: "Turnaround Finger Roll",
    63: "Fadeaway Jumper",
    64: "Follow Up Dunk",
    65: "Jump Hook Shot",
    66: "Jump Bank Shot",
    67: "Hook Bank Shot",
    71: "Finger Roll Layup",
    72: "Putback Layup",
    73: "Driving Reverse Layup",
    74: "Running Reverse Layup",
    75: "Driving Finger Roll Layup",
    76: "Running Finger Roll Layup",
    77: "Driving Jump Shot",
    78: "Floating Jump Shot",
    79: "Pullup Jump Shot",
    80: "Step Back Jump Shot",
    81: "Pullup Bank Shot",
    82: "Driving Bank Shot",
    83: "Fadeaway Bank Shot",
    84: "Running Bank Shot",
    85: "Turnaround Bank Shot",
    86: "Turnaround Fadeaway",
    87: "Putback Dunk",
    88: "Driving Slam Dunk",
    89: "Reverse Slam Dunk",
    90: "Running Slam Dunk",
    91: "Putback Reverse Dunk",
    92: "Putback Slam Dunk",
    93: "Driving Bank Hook Shot",
    94: "Jump Bank Hook Shot",
    95: "Running Bank Hook Shot",
    96: "Turnaround Bank Hook Shot",
    97: "Tip Layup Shot",
    98: "Cutting Layup Shot",
    99: "Cutting Finger Roll Layup Shot",
    100: "Running Alley Oop Layup Shot",
    101: "Driving Floating Jump Shot",
    102: "Driving Floating Bank Jump Shot",
    103: "Running Pull-Up Jump Shot",
    104: "Step Back Bank Jump Shot",
    105: "Turnaround Fadeaway Bank Jump Shot",
    106: "Running Alley Oop Dunk Shot",
    107: "Tip Dunk Shot",
    108: "Cutting Dunk Shot",
    109: "Driving Reverse Dunk Shot",
    110: "Running Reverse Dunk Shot"
}

EVENTMSGACTIONTYPE_FREE_THROW_dict = {
    10: "Free Throw 1 of 1",
    11: "Free Throw 1 of 2",
    12: "Free Throw 2 of 2",
    13: "Free Throw 1 of 3",
    14: "Free Throw 2 of 3",
    15: "Free Throw 3 of 3",
    16: "Free Throw Technical",
    17: "Free Throw Clear Path",  # before 2006-07 season
    18: "Free Throw Flagrant 1 of 2",
    19: "Free Throw Flagrant 2 of 2",
    20: "Free Throw Flagrant 1 of 1",
    21: "Free Throw Technical 1 of 2",
    22: "Free Throw Technical 2 of 2",
    23: "Free Throw 1 of 2",  # Guess
    24: "Free Throw 2 of 2",  # Guess
    25: "Free Throw Clear Path 1 of 2",
    26: "Free Throw Clear Path 2 of 2",
    27: "Free Throw Flagrant 1 of 3",
    28: "Free Throw Flagrant 2 of 3",
    29: "Free Throw Flagrant 3 of 3"
}

EVENTMSGACTIONTYPE_REBOUND_dict = {
    0: "live", #Rebound from live game. After shot, ball out of bounds, etc.
    1: "pause" #After miss of non-final free throw. Example: A player is awarded two free throws and misses the first. He gets the ball for the second attempt without the game resuming ('rebounds' to ball).
}

# TODO - add first column to columns, first column is the sequence in game
column_names = ["Unnamed: 0", "EVENTMSGACTIONTYPE", "EVENTMSGTYPE", "EVENTNUM",
                "GAME_ID", "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "PCTIMESTRING",
                "PERIOD", "PERSON1TYPE", "PERSON2TYPE", "PERSON3TYPE", "PLAYER1_ID",
                "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION", "PLAYER1_TEAM_CITY",
                "PLAYER1_TEAM_ID", "PLAYER1_TEAM_NICKNAME", "PLAYER2_ID",
                "PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION", "PLAYER2_TEAM_CITY",
                "PLAYER2_TEAM_ID", "PLAYER2_TEAM_NICKNAME", "PLAYER3_ID",
                "PLAYER3_NAME", "PLAYER3_TEAM_ABBREVIATION", "PLAYER3_TEAM_CITY",
                "PLAYER3_TEAM_ID", "PLAYER3_TEAM_NICKNAME", "SCORE", "SCOREMARGIN",
                "VISITORDESCRIPTION", "WCTIMESTRING"]

dtype = {"PLAYER1_TEAM_ID": str, "PLAYER2_TEAM_ID": str, "PLAYER3_TEAM_ID": str, "PLAYER1_ID": str, "PLAYER2_ID":str, "PLAYER3_ID": str}


def load_data(columns=None, seasons=None, path=RAW_DATA_PATH, resolve=True, single_df = True):
    """
    TODO - add docstring
    :param columns:
    :param seasons:
    :param path:
    :param resolve:
    :return:
    """
    # TODO - check could be added if the sequence of the actions matches timestamps
    E_ACT_TYPE_STR = "EVENTMSGACTIONTYPE"
    E_TYPE_STR = "EVENTMSGTYPE"
    P_TYPE_STR_LIST = [f"PERSON{i}TYPE" for i in [1, 2, 3]] #For Person3 there seem to be some random 1 : Timeout

    files = sorted([f for f in path.glob("*.csv")])
    season_dfs = []
    for file_name in files:
        # Only read specified seasons:
        if seasons and (file_name.stem[:7] not in seasons):
            continue
        season = pd.read_csv(file_name, usecols=columns, dtype=dtype)

        season["season_name"] = file_name.stem[:7]
        if resolve:
            if columns is None or E_TYPE_STR in columns:
                season[E_TYPE_STR] = season[E_TYPE_STR].replace(EVENTMSGTYPE_dict)
            if columns is None or E_ACT_TYPE_STR in columns:
                mask_field_goal_made = season[E_TYPE_STR] == EVENTMSGTYPE_dict[1]
                mask_field_goal_miss = season[E_TYPE_STR] == EVENTMSGTYPE_dict[2]
                mask_free_throw = season[E_TYPE_STR] == EVENTMSGTYPE_dict[3]
                mask_rebound = season[E_TYPE_STR] == EVENTMSGTYPE_dict[4]

                season.loc[mask_field_goal_made, E_ACT_TYPE_STR] = season.loc[mask_field_goal_made, E_ACT_TYPE_STR]\
                    .replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
                season.loc[mask_field_goal_miss, E_ACT_TYPE_STR] = season.loc[mask_field_goal_miss, E_ACT_TYPE_STR]\
                    .replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
                season.loc[mask_free_throw, E_ACT_TYPE_STR] = season.loc[mask_free_throw, E_ACT_TYPE_STR]\
                    .replace(EVENTMSGACTIONTYPE_FREE_THROW_dict)
                season.loc[mask_rebound, E_ACT_TYPE_STR] = season.loc[mask_rebound, E_ACT_TYPE_STR]\
                    .replace(EVENTMSGACTIONTYPE_REBOUND_dict)
            for P_TYPE_STR in P_TYPE_STR_LIST:
                if columns is None or P_TYPE_STR in columns:
                    season[P_TYPE_STR] = season[P_TYPE_STR].replace(PERSONTYPE_dict)

        # remove empty columns:
        if "NEUTRALDESCRIPTION" in season.columns:
            season = season.drop("NEUTRALDESCRIPTION", axis=1)
        season_dfs.append(season)
    if single_df:
        return pd.concat(season_dfs).reset_index(drop=True) if season_dfs else None
    else:
        return season_dfs

def load_game_data(columns=None, seasons=None, path=RAW_DATA_PATH):
    GAME_ID_STR = "GAME_ID"
    
    pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False)
    print("Loaded PBP-data")
    game_data_per_season = []
    for pbp_data in pbp_data_per_season:
        pbp_grouped = pbp_data.groupby(GAME_ID_STR)

        #Create DataFrame with first column "play_count"
        game_data = pbp_grouped.size().to_frame(name="play_count")
        #Add arbitrarily many features:

        #Season name is the same for all plays in a game: Just get first.
        game_data["season_name"] = pbp_grouped["season_name"].first()
        #Get date and start/end time would be nice.

        #Home team:
        game_data["visitor_team_id"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
        game_data["visitor_team_city"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
        game_data["visitor_team_nickname"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])
        #Score:
        game_data[["visitor_final_score", "home_final_score"]] = pbp_grouped.apply(lambda x: x[~x["SCORE"].isna()]["SCORE"].str.split(" - ", expand =True).astype(int).max()) #Complicated because in ca. 5 games the score column is messed up and out of order.
        #Visitor team:
        game_data["home_team_id"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
        game_data["home_team_city"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
        game_data["home_team_nickname"] = pbp_grouped.apply(lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])

        #Number of periods played: >4 is overtime
        game_data["periods"] = pbp_grouped["PERIOD"].max()

        #Field goal stats
        game_data["visitor_fg_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["visitor_fg_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["visitor_3PT_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("3PT"))).sum(level=0)
        game_data["visitor_3PT_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("3PT"))).sum(level=0)
        game_data["home_fg_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)
        game_data["home_fg_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)
        game_data["home_3PT_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("3PT"))).sum(level=0)
        game_data["home_3PT_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("3PT"))).sum(level=0)

        #Free throw stats
        game_data["visitor_ft_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("MISS") == False)).sum(level=0)
        game_data["visitor_ft_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("MISS"))).sum(level=0)
        game_data["home_ft_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("MISS") == False)).sum(level=0)
        game_data["home_ft_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("MISS"))).sum(level=0)
        
        #Rebound stats:
        #Player rebounds the ball in live game
        game_data["visitor_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["EVENTMSGACTIONTYPE"] == "live")).sum(level=0)
        game_data["home_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["EVENTMSGACTIONTYPE"] == "live")).sum(level=0)
        #Team gets the ball if out of bounds
        game_data["visitor_team_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_TEAM') & (x["EVENTMSGACTIONTYPE"] == "live")).sum(level=0)
        game_data["home_team_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_TEAM') & (x["EVENTMSGACTIONTYPE"] == "live")).sum(level=0)

        #Turnover stats:
        #Player turns the ball over: bad pass, offensive foul, ...
        game_data["visitor_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["home_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)
        #Team turn over: shot clock violation, 5 sec violation
        game_data["visitor_team_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).sum(level=0)
        game_data["home_team_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_TEAM')).sum(level=0)

        #Foul stats:
        game_data["visitor_foul"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["home_foul"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)

        #Substitution stats:
        game_data["visitor_subs"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["home_subs"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)

        #Timeout stats:
        game_data["visitor_timeout"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).sum(level=0)
        game_data["home_timeout"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'HOME_TEAM')).sum(level=0)

        #Jump ball stats:
        game_data["visitor_jump_balls_won"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["home_jump_balls_won"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'HOME_PLAYER')).sum(level=0)
        game_data["tip_off_winner"] = pbp_grouped.apply(lambda x: x[x["EVENTMSGTYPE"] == 'JUMP_BALL']["PERSON3TYPE"].iloc[0] if 'JUMP_BALL' in x["EVENTMSGTYPE"].unique() else "UNKNOWN")

        #Ejection stats:
        #Player on the court gets ejected
        game_data["visitor_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).sum(level=0)
        game_data["home_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).sum(level=0)
        #Non player gets ejected: coach etc.
        game_data["visitor_team_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_TEAM_FOUL')).sum(level=0)
        game_data["home_team_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_TEAM_FOUL')).sum(level=0)

        game_data_per_season.append(game_data)
        print(f"Calculated game data for the {pbp_data['season_name'][0]} season", end = "\r")
    
    return pd.concat(game_data_per_season) if game_data_per_season else None