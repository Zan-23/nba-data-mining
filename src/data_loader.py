from copy import deepcopy

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
PLAYER_INFO_PATH = Path(__file__).parent.parent / "data" / "raw" / "player-data" / "player_info.csv"

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
    2: "HOME_TEAM",  # If the play is not attributed to a single player
    3: "VISITOR_TEAM",  # If the play is not attributed to a single player
    4: "HOME_PLAYER",  # If the play is not attributed to a single player
    5: "VISITOR_PLAYER",  # If the play is not attributed to a single player
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
    0: "live",  # Rebound from live game. After shot, ball out of bounds, etc.
    1: "pause"
    # After miss of non-final free throw. Example: A player is awarded two free throws and misses the first. He gets the ball for the second attempt without the game resuming ('rebounds' to ball).
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

dtype = {"PLAYER1_TEAM_ID": str, "PLAYER2_TEAM_ID": str, "PLAYER3_TEAM_ID": str, "PLAYER1_ID": str, "PLAYER2_ID": str,
         "PLAYER3_ID": str}


def load_data(columns=None, seasons=None, path=RAW_DATA_PATH, resolve=True, single_df=True):
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
    P_TYPE_STR_LIST = [f"PERSON{i}TYPE" for i in [1, 2, 3]]  # For Person3 there seem to be some random 1 : Timeout

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

                season.loc[mask_field_goal_made, E_ACT_TYPE_STR] = season.loc[mask_field_goal_made, E_ACT_TYPE_STR] \
                    .replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
                season.loc[mask_field_goal_miss, E_ACT_TYPE_STR] = season.loc[mask_field_goal_miss, E_ACT_TYPE_STR] \
                    .replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
                season.loc[mask_free_throw, E_ACT_TYPE_STR] = season.loc[mask_free_throw, E_ACT_TYPE_STR] \
                    .replace(EVENTMSGACTIONTYPE_FREE_THROW_dict)
                season.loc[mask_rebound, E_ACT_TYPE_STR] = season.loc[mask_rebound, E_ACT_TYPE_STR] \
                    .replace(EVENTMSGACTIONTYPE_REBOUND_dict)
            for P_TYPE_STR in P_TYPE_STR_LIST:
                if columns is None or P_TYPE_STR in columns:
                    season[P_TYPE_STR] = season[P_TYPE_STR].replace(PERSONTYPE_dict)

        # remove empty columns:
        if "NEUTRALDESCRIPTION" in season.columns:
            season = season.drop("NEUTRALDESCRIPTION", axis=1)
        season_dfs.append(season)

    [get_distance(df) for df in season_dfs]

    if single_df:
        return pd.concat(season_dfs).reset_index(drop=True) if season_dfs else None
    else:
        return season_dfs


def get_distance(df):
    df["home_shot_distance"] = df['HOMEDESCRIPTION'].str.extract("(\d{1,2}\')")
    df["visitor_shot_distance"] = df['VISITORDESCRIPTION'].str.extract("(\d{1,2}\')")

    df['home_shot_distance'] = df['home_shot_distance'].str.replace('\'', '').astype(float)
    #df['home_shot_distance'] = df['home_shot_distance'].fillna(-1).astype(int)
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'HOME_PLAYER') & (df['home_shot_distance'].isna()) & (df["HOMEDESCRIPTION"].str.contains("3PT ")), 'home_shot_distance'] = 23
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'HOME_PLAYER') & (df['home_shot_distance'].isna()), 'home_shot_distance'] = 0

    df['visitor_shot_distance'] = df['visitor_shot_distance'].str.replace('\'', '').astype(float)
    #df['visitor_shot_distance'] = df['visitor_shot_distance'].fillna(-1).astype(int)
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'VISITOR_PLAYER') & (df['visitor_shot_distance'].isna()) & (df["VISITORDESCRIPTION"].str.contains("3PT ")), 'visitor_shot_distance'] = 23
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'VISITOR_PLAYER') & (df['visitor_shot_distance'].isna()), 'visitor_shot_distance'] = 0


def load_game_data(columns=None, seasons=None, path=RAW_DATA_PATH):
    GAME_ID_STR = "GAME_ID"

    pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False)
    print("Loaded PBP-data")
    game_data_per_season = []
    for pbp_data in pbp_data_per_season:
        pbp_grouped = pbp_data.groupby(GAME_ID_STR)

        # Create DataFrame with first column "play_count"
        game_data = pbp_grouped.size().to_frame(name="play_count")
        # Add arbitrarily many features:

        # Season name is the same for all plays in a game: Just get first.
        game_data["season_name"] = pbp_grouped["season_name"].first()
        # Get date and start/end time would be nice.

        # Visitor team:
        game_data["visitor_team_id"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
        game_data["visitor_team_city"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
        game_data["visitor_team_nickname"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])
        game_data["visitor_record_wins"] = 0
        game_data["visitor_record_losses"] = 0
        # Score:
        game_data[["visitor_final_score", "home_final_score"]] = pbp_grouped.apply(
            lambda x: x[~x["SCORE"].isna()]["SCORE"].str.split(" - ", expand=True).astype(
                int).max())  # Complicated because in ca. 5 games the score column is messed up and out of order.
        game_data["home_win"] = game_data["visitor_final_score"] < game_data["home_final_score"]
        # Home team:
        game_data["home_team_id"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
        game_data["home_team_city"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
        game_data["home_team_nickname"] = pbp_grouped.apply(
            lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])
        # Calculated later
        game_data["home_record_wins"] = 0
        game_data["home_record_losses"] = 0

        # Number of periods played: >4 is overtime
        game_data["periods"] = pbp_grouped["PERIOD"].max()
        #Minutes played:
        game_data["minutes_played"] = 48 + (game_data["periods"]-4) * 2.5

        #Players deployed
        game_data["visitor_players_deployed"] = pbp_grouped.apply(lambda x: len(set.union(*[set(x[x[f"PERSON{i}TYPE"]=="VISITOR_PLAYER"][f"PLAYER{i}_ID"].unique()) for i in range(1,4)])))
        game_data["home_players_deployed"] = pbp_grouped.apply(lambda x: len(set.union(*[set(x[x[f"PERSON{i}TYPE"]=="HOME_PLAYER"][f"PLAYER{i}_ID"].unique()) for i in range(1,4)])))

        #Field goal stats
        game_data["visitor_fg_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["visitor_fg_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["visitor_3PT_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
        game_data["visitor_3PT_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
        game_data["home_fg_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
        game_data["home_fg_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
        game_data["home_3PT_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
        game_data["home_3PT_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()

        #Free throw stats
        game_data["visitor_ft_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("MISS") == False)).groupby(level=0).sum()
        game_data["visitor_ft_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["VISITORDESCRIPTION"].str.contains("MISS"))).groupby(level=0).sum()
        game_data["home_ft_made"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("MISS") == False)).groupby(level=0).sum()
        game_data["home_ft_missed"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["HOMEDESCRIPTION"].str.contains("MISS"))).groupby(level=0).sum()
        
        #Rebound stats:
        #Player rebounds the ball in live game
        game_data["visitor_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
        game_data["home_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
        #Team gets the ball if out of bounds
        game_data["visitor_team_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_TEAM') & (x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
        game_data["home_team_rebound"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_TEAM') & (x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()

        #Turnover stats:
        #Player turns the ball over: bad pass, offensive foul, ...
        game_data["visitor_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["home_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
        #Team turn over: shot clock violation, 5 sec violation
        game_data["visitor_team_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).groupby(level=0).sum()
        game_data["home_team_turnover"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_TEAM')).groupby(level=0).sum()

        #Foul stats:
        game_data["visitor_foul"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["home_foul"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()

        #Substitution stats:
        game_data["visitor_subs"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["home_subs"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()

        #Timeout stats:
        game_data["visitor_timeout"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).groupby(level=0).sum()
        game_data["home_timeout"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'HOME_TEAM')).groupby(level=0).sum()

        #Jump ball stats:
        game_data["visitor_jump_balls_won"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["home_jump_balls_won"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
        game_data["tip_off_winner"] = pbp_grouped.apply(lambda x: x[x["EVENTMSGTYPE"] == 'JUMP_BALL']["PERSON3TYPE"].iloc[0] if 'JUMP_BALL' in x["EVENTMSGTYPE"].unique() else "UNKNOWN")

        #Ejection stats:
        #Player on the court gets ejected
        game_data["visitor_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
        game_data["home_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
        #Non player gets ejected: coach etc.
        game_data["visitor_team_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_TEAM_FOUL')).groupby(level=0).sum()
        game_data["home_team_ejection"] = pbp_grouped.apply(lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_TEAM_FOUL')).groupby(level=0).sum()

        #Player performance:
        #Points of scoring leader
        game_data["home_scoring_leader"] = pbp_grouped.apply(lambda x: (
            (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["HOMEDESCRIPTION"].str.contains('3PT')) & (
                        x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
                x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & ((x["HOMEDESCRIPTION"].str.contains('3PT')) == False) & (
                            x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 2, fill_value=0).add(
                x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["HOMEDESCRIPTION"].str.contains('MISS')) == False) & (
                            x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
        ).idxmax())
        game_data["home_scoring_leader_points"] = pbp_grouped.apply(lambda x: (
            (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["HOMEDESCRIPTION"].str.contains('3PT')) & (
                        x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
                x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & ((x["HOMEDESCRIPTION"].str.contains('3PT')) == False) & (
                            x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 2, fill_value=0).add(
                x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["HOMEDESCRIPTION"].str.contains('MISS')) == False) & (
                            x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
        ).max())
        game_data["visitor_scoring_leader"] = pbp_grouped.apply(lambda x: (
            (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["VISITORDESCRIPTION"].str.contains('3PT')) & (
                        x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
                x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
                            (x["VISITORDESCRIPTION"].str.contains('3PT')) == False) & (
                              x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 2,
                fill_value=0).add(
                x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["VISITORDESCRIPTION"].str.contains('MISS')) == False) & (
                            x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
        ).idxmax())
        game_data["visitor_scoring_leader_points"] = pbp_grouped.apply(lambda x: (
            (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["VISITORDESCRIPTION"].str.contains('3PT')) & (
                        x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
                x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
                            (x["VISITORDESCRIPTION"].str.contains('3PT')) == False) & (
                              x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 2,
                fill_value=0).add(
                x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["VISITORDESCRIPTION"].str.contains('MISS')) == False) & (
                            x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
        ).max())

        # Shooting Distance information
        # Max and min distance
        game_data["home_made_max_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].max())
        game_data["visitor_made_max_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].max())

        game_data["home_made_min_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].min())
        game_data["visitor_made_min_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].min())

        game_data["home_made_mean_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].mean())
        game_data["visitor_made_mean_shot_distance"] = pbp_grouped.apply(
            lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].mean())

        #Calculate record after game for both teams (total number of wins and losses for the season):
        wins_dict = {team: 0 for team in set(game_data["visitor_team_id"].to_list() + game_data["home_team_id"].to_list())}
        losses_dict = wins_dict.copy()
        for i, row in game_data.iterrows():
            if row["home_win"]:
                wins_dict[row["home_team_id"]] += 1
                losses_dict[row["visitor_team_id"]] += 1
            else:
                wins_dict[row["visitor_team_id"]] += 1
                losses_dict[row["home_team_id"]] += 1
            game_data.at[i,"home_record_wins"] = wins_dict[row["home_team_id"]]
            game_data.at[i,"home_record_losses"] = losses_dict[row["home_team_id"]]
            game_data.at[i,"visitor_record_wins"] = wins_dict[row["visitor_team_id"]]
            game_data.at[i,"visitor_record_losses"] = losses_dict[row["visitor_team_id"]]

        game_data_per_season.append(game_data)
        print(f"Calculated game data for the {pbp_data['season_name'][0]} season", end="\r")

    return pd.concat(game_data_per_season) if game_data_per_season else None


def load_player_data(columns=None, seasons=None, path=RAW_DATA_PATH, player_info_path=PLAYER_INFO_PATH):
    pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False)
    print("Loaded PBP-data")
    player_info = pd.read_csv(PLAYER_INFO_PATH, dtype={"Player ID": str})
    print("Loaded player-info data")
    player_data_per_season = []
    for pbp_data in pbp_data_per_season:
        # Filter fauly events
        pbp_data = pbp_data[pbp_data["EVENTMSGTYPE"] != 18]
        # Perform groupby only once
        pbp_grouped_pID1 = pbp_data.groupby("PLAYER1_ID")
        pbp_grouped_pID2 = pbp_data.groupby("PLAYER2_ID")
        pbp_grouped_pID3 = pbp_data.groupby("PLAYER3_ID")

        # Create row for each player involved in at least one play.
        player_data = pd.DataFrame(index=set.union(*[
            set(pbp_data[pbp_data[f"PERSON{i}TYPE"].isin(["HOME_PLAYER", "VISITOR_PLAYER"])][f"PLAYER{i}_ID"].unique())
            for i in range(1, 4)]))

        # Get player name:
        player_data["season_name"] = pbp_data.iloc[0]["season_name"]
        player_data["player_name1"] = pbp_grouped_pID1["PLAYER1_NAME"].first()
        player_data["player_name2"] = pbp_grouped_pID2["PLAYER2_NAME"].first()
        player_data["player_name3"] = pbp_grouped_pID3["PLAYER3_NAME"].first()
        player_data['player_name'] = player_data[['player_name1', 'player_name2', 'player_name3']].fillna(
            method='bfill', axis=1).iloc[:, 0]
        player_data = player_data.drop(columns=['player_name1', 'player_name2', 'player_name3'])
        player_data = player_data[player_data[
            'player_name'].notna()]  # Removes ids for coaches getting ejections/fouls in 2017-18 season while being listed as player. (they dont have name)

        # Add scraped player-info
        player_data = player_data.merge(
            player_info[player_info["Season"] == pbp_data.iloc[0]["season_name"]].set_index("Player ID")[
                ["Age", "Height", "Weight", "College", "Country", "Draft Year", "Draft Number", "Draft Number"]],
            left_index=True, right_index=True)

        # Get games played:
        player_deployed_per_game = pbp_data.groupby("GAME_ID").apply(lambda x: set.union(
            *[set(x[x[f"PERSON{i}TYPE"].isin(["HOME_PLAYER", "VISITOR_PLAYER"])][f"PLAYER{i}_ID"].unique()) for i in
              range(1, 4)]))
        player_data["games_played"] = player_data.apply(
            lambda x: len([game for game in player_deployed_per_game if x.name in game]), axis=1)

        # Field goal stats:
        player_data["fg_made"] = pbp_grouped_pID1.apply(
            lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PLAYER1_ID"] == x.name)).groupby(level=0).sum()
        player_data["fg_made"] = player_data["fg_made"].fillna(0).astype(int)
        player_data["fg_missed"] = pbp_grouped_pID1.apply(
            lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PLAYER1_ID"] == x.name)).groupby(level=0).sum()
        player_data["fg_missed"] = player_data["fg_missed"].fillna(0).astype(int)
        player_data["3PT_made"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PLAYER1_ID"] == x.name)) & (
                    (x["VISITORDESCRIPTION"].str.contains("3PT").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("3PT").fillna(False)))).groupby(level=0).sum()
        player_data["3PT_made"] = player_data["3PT_made"].fillna(0).astype(int)
        player_data["3PT_missed"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PLAYER1_ID"] == x.name)) & (
                    (x["VISITORDESCRIPTION"].str.contains("3PT").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("3PT").fillna(False)))).groupby(level=0).sum()
        player_data["3PT_missed"] = player_data["3PT_missed"].fillna(0).astype(int)

        # Free throw stats:
        player_data["ft_made"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PLAYER1_ID"] == x.name)) & (~(
                    (x["VISITORDESCRIPTION"].str.contains("MISS").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("MISS").fillna(False))))).groupby(level=0).sum()
        player_data["ft_made"] = player_data["ft_made"].fillna(0).astype(int)
        player_data["ft_missed"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PLAYER1_ID"] == x.name)) & (
                    (x["VISITORDESCRIPTION"].str.contains("MISS").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("MISS").fillna(False)))).groupby(level=0).sum()
        player_data["ft_missed"] = player_data["ft_missed"].fillna(0).astype(int)

        # Points scored:
        player_data["points"] = 3 * player_data["3PT_made"] + 2 * (player_data["fg_made"] - player_data["3PT_made"]) + \
                                player_data["ft_made"]

        # Rebound stats:
        player_data["rebounds"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'REBOUND') & (x["PLAYER1_ID"] == x.name))).groupby(level=0).sum()
        player_data["rebounds"] = player_data["rebounds"].fillna(0).astype(int)

        # Assist stats:
        player_data["assists"] = pbp_grouped_pID2.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PLAYER2_ID"] == x.name))).groupby(level=0).sum()
        player_data["assists"] = player_data["assists"].fillna(0).astype(int)

        # Turnover stats:
        player_data["turnover"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PLAYER1_ID"] == x.name))).groupby(level=0).sum()
        player_data["turnover"] = player_data["turnover"].fillna(0).astype(int)

        # Foul stats: (not just personal fouls)
        player_data["fouls"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FOUL') & (x["PLAYER1_ID"] == x.name))).groupby(level=0).sum()
        player_data["fouls"] = player_data["fouls"].fillna(0).astype(int)

        player_data_per_season.append(player_data)
        print(f"Calculated player data for the {pbp_data['season_name'][0]} season", end="\r")
    return pd.concat(player_data_per_season).sort_values(
        ["player_name", "season_name"]) if player_data_per_season else None


def load_player_data_optimized(columns=None, seasons=None, path=RAW_DATA_PATH, player_info_path=PLAYER_INFO_PATH):
    # TODO still have to finish optimization, then replace it, or add reading from preprocessed file
    pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False)
    print("Loaded PBP-data")
    player_info = pd.read_csv(PLAYER_INFO_PATH, dtype={"Player ID": str})
    print("Loaded player-info data")

    player_data_per_season = []
    for pbp_data in pbp_data_per_season:
        # Filter fauly events
        pbp_data = pbp_data[pbp_data["EVENTMSGTYPE"] != 18]
        # Perform groupby only once
        pbp_grouped_pID1 = pbp_data.groupby("PLAYER1_ID")
        pbp_grouped_pID2 = pbp_data.groupby("PLAYER2_ID")
        pbp_grouped_pID3 = pbp_data.groupby("PLAYER3_ID")

        # Create row for each player involved in at least one play.
        player_data = pd.DataFrame(index=set.union(*[
            set(pbp_data[pbp_data[f"PERSON{i}TYPE"].isin(["HOME_PLAYER", "VISITOR_PLAYER"])][f"PLAYER{i}_ID"].unique())
            for i in range(1, 4)]))

        # Get player name:
        player_data["season_name"] = pbp_data.iloc[0]["season_name"]
        player_data["player_name1"] = pbp_grouped_pID1["PLAYER1_NAME"].first()
        player_data["player_name2"] = pbp_grouped_pID2["PLAYER2_NAME"].first()
        player_data["player_name3"] = pbp_grouped_pID3["PLAYER3_NAME"].first()
        player_data['player_name'] = player_data[['player_name1', 'player_name2', 'player_name3']].fillna(
            method='bfill', axis=1).iloc[:, 0]
        player_data = player_data.drop(columns=['player_name1', 'player_name2', 'player_name3'])
        # Removes ids for coaches getting ejections/fouls in 2017-18 season while being listed as player. (they dont have name)
        player_data = player_data[player_data['player_name'].notna()]

        # Add scraped player-info
        player_data = player_data.merge(
            player_info[player_info["Season"] == pbp_data.iloc[0]["season_name"]].set_index("Player ID")[
                ["Age", "Height", "Weight", "College", "Country", "Draft Year", "Draft Number", "Draft Number"]],
            left_index=True, right_index=True)

        # Get games played:
        player_deployed_per_game = pbp_data.groupby("GAME_ID").apply(lambda x: set.union(
            *[set(x[x[f"PERSON{i}TYPE"].isin(["HOME_PLAYER", "VISITOR_PLAYER"])][f"PLAYER{i}_ID"].unique()) for i in
              range(1, 4)]))
        player_data["games_played"] = player_data.apply(
            lambda x: len([game for game in player_deployed_per_game if x.name in game]), axis=1)
        player_data["games_played"] = player_data["games_played"].fillna(0).astype(int)

        # Field goal stats:
        player_data["fg_made"] = pbp_grouped_pID1.apply(
            lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PLAYER1_ID"] == x.name)).groupby(level=0).sum()
        player_data["fg_made"] = player_data["fg_made"].fillna(0).astype(int)
        player_data["fg_missed"] = pbp_grouped_pID1.apply(
            lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PLAYER1_ID"] == x.name)).groupby(level=0).sum()
        player_data["fg_missed"] = player_data["fg_missed"].fillna(0).astype(int)
        player_data["3PT_made"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PLAYER1_ID"] == x.name)) & (
                    (x["VISITORDESCRIPTION"].str.contains("3PT").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("3PT").fillna(False)))).groupby(level=0).sum()
        player_data["3PT_made"] = player_data["3PT_made"].fillna(0).astype(int)
        player_data["3PT_missed"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PLAYER1_ID"] == x.name)) & (
                    (x["VISITORDESCRIPTION"].str.contains("3PT").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("3PT").fillna(False)))).groupby(level=0).sum()
        player_data["3PT_missed"] = player_data["3PT_missed"].fillna(0).astype(int)

        # Free throw stats:
        # TODO why is only one negated?
        mask_desc_miss = [(pbp_grouped_pID1["VISITORDESCRIPTION"].str.contains("MISS").fillna(False))
                          | (pbp_grouped_pID1["HOMEDESCRIPTION"].str.contains("MISS").fillna(False))]
        player_data["ft_made"] = pbp_grouped_pID1[(pbp_grouped_pID1["EVENTMSGTYPE"] == 'FREE_THROW') &
                                                  (~mask_desc_miss)].groupby(level=0).sum()
        player_data["ft_made"] = pbp_grouped_pID1.apply(
            lambda x: ((x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PLAYER1_ID"] == x.name)) & (~(
                    (x["VISITORDESCRIPTION"].str.contains("MISS").fillna(False)) | (
                x["HOMEDESCRIPTION"].str.contains("MISS").fillna(False))))).groupby(level=0).sum()
        player_data["ft_made"] = player_data["ft_made"].fillna(0).astype(int)

        # event type has to be be free throw and
        player_data["ft_missed"] = pbp_grouped_pID1[(pbp_grouped_pID1["EVENTMSGTYPE"] == 'FREE_THROW') &
                                                    (mask_desc_miss)].groupby(level=0).sum()
        player_data["ft_missed"] = player_data["ft_missed"].fillna(0).astype(int)

        # Points scored:
        player_data["points"] = 3 * player_data["3PT_made"] + 2 * (player_data["fg_made"] - player_data["3PT_made"]) + \
                                player_data["ft_made"]

        # Rebound stats:
        player_data["rebounds"] = pbp_grouped_pID1[pbp_grouped_pID1["EVENTMSGTYPE"] == "REBOUND"].groupby(level=0).sum()
        player_data["rebounds"] = player_data["rebounds"].fillna(0).astype(int)

        # Assist stats:
        player_data["assists"] = pbp_grouped_pID1[pbp_grouped_pID1["EVENTMSGTYPE"] == "FIELD_GOAL_MADE"].groupby(
            level=0).sum()
        player_data["assists"] = player_data["assists"].fillna(0).astype(int)

        # Turnover stats:
        player_data["turnover"] = pbp_grouped_pID1[pbp_grouped_pID1["EVENTMSGTYPE"] == "TURNOVER"].groupby(
            level=0).sum()
        player_data["turnover"] = player_data["turnover"].fillna(0).astype(int)

        # Foul stats: (not just personal fouls)
        player_data["fouls"] = pbp_grouped_pID1[(pbp_grouped_pID1["EVENTMSGTYPE"] == "FOUL")].groupby(level=0).sum()
        player_data["fouls"] = player_data["fouls"].fillna(0).astype(int)

        player_data_per_season.append(player_data)
        print(f"Calculated player data for the {pbp_data['season_name'][0]} season", end="\r")
    return pd.concat(player_data_per_season).sort_values(
        ["player_name", "season_name"]) if player_data_per_season else None


def player_data_preprocessing(player_df, remove_nan_rows=False):
    """
    Preprocesses the player data.

    :param player_df: Dataframe with player data.
    :param remove_nan_rows: Boolean. If True, rows with NaN values are removed.
    :return: A modified dataframe with player data standardized to kg and cm.
    """
    pounds_to_kg_ratio = 0.45359237
    feet_to_cm_ratio = 30.48
    inches_to_cm_ratio = 2.54

    preprocessed_df = deepcopy(player_df)
    #  pounds to kg
    preprocessed_df["Weight"] = preprocessed_df["Weight"].replace("-", np.nan)
    preprocessed_df["Weight"] = (preprocessed_df["Weight"].astype(float) * pounds_to_kg_ratio).astype(float)

    # feet to cm
    split_data = preprocessed_df["Height"].str.split('-').apply(pd.Series).replace("", np.nan).astype(float)
    preprocessed_df["Height"] = (split_data[0] * feet_to_cm_ratio + split_data[1] * inches_to_cm_ratio).astype(float)

    if remove_nan_rows:
        # remove rows where height or data is zero since it skews the result
        tmp_len = len(preprocessed_df.index)

        preprocessed_df = preprocessed_df[(preprocessed_df["Height"] != np.nan) | (preprocessed_df["Weight"] != np.nan)]
        print(f"Removed {tmp_len - len(player_df.index)} rows with 0/nan values!")

    return preprocessed_df

