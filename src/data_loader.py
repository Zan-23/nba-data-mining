import os
import pickle
from collections import defaultdict
from copy import deepcopy
from src.data_specification import LOAD_DATA_COL_TYPES, CATEGORIES_COLS_ARR
from src.data_specification import EVENTMSGTYPE_dict, EVENTMSGACTIONTYPE_FIELD_GOAL_dict, \
    EVENTMSGACTIONTYPE_FREE_THROW_dict, EVENTMSGACTIONTYPE_REBOUND_dict, PERSONTYPE_dict

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
PLAYER_INFO_PATH = Path(__file__).parent.parent / "data" / "raw" / "player-data" / "player_info.csv"


def load_data(columns=None, seasons=None, path=RAW_DATA_PATH, resolve=True, single_df=True, force_recompute=True):
    """
    Function computes a combined dataframe or an array of dataframes for all seasons or just the provide ones.

    :param columns:
    :param seasons:
    :param path:
    :param resolve:
    :param single_df:
    :param force_recompute: Boolean. If true the data is recomputed else it is loaded from the pickle file if it
    exists.
    :return:
    """
    E_ACT_TYPE_STR = "EVENTMSGACTIONTYPE"
    E_TYPE_STR = "EVENTMSGTYPE"
    P_TYPE_STR_LIST = [f"PERSON{i}TYPE" for i in [1, 2, 3]]  # For Person3 there seem to be some random 1 : Timeout

    dir_path = os.path.dirname(os.path.realpath(__file__))
    season_arr_file_path = os.path.join(dir_path, "./../data/processed/load_data_seasons_arr.pkl")

    # reading from file or recomputing depending on the flags
    if not force_recompute and os.path.exists(season_arr_file_path):
        print(f"Loading data from file {season_arr_file_path} ...")
        with open(season_arr_file_path, "rb") as file:
            seasons_arr = pickle.load(file)

        if single_df:
            # return a concatenated dataframe
            return pd.concat(seasons_arr).reset_index(drop=True)
        else:
            return seasons_arr
    else:
        print("Recomputing data ...")
        files = sorted([f for f in path.glob("*.csv")])
        seasons_df_arr = []
        for file_name in files:
            # Only read specified seasons:
            if seasons and (file_name.stem[:7] not in seasons):
                continue
            season = pd.read_csv(file_name, usecols=columns)

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

            # modifies dataframe in place
            get_distance(season)
            # modifies type in place to be a category
            # TODO comment it out if you want to use categories
            # season[CATEGORIES_COLS_ARR] = season[CATEGORIES_COLS_ARR].astype("category")
            seasons_df_arr.append(season)

        if len(seasons_df_arr) == 0:
            raise Exception("Season dfs is empty! Check if the season names are valid!")

        # saving seasons arr to file, can be recomputed as single df
        with open(season_arr_file_path, "wb") as file:
            pickle.dump(seasons_df_arr, file)

        if single_df:
            concat_df = pd.concat(seasons_df_arr).reset_index(drop=True)
            # TODO comment it out if you want to use categories
            # concat_df[CATEGORIES_COLS_ARR] = concat_df[CATEGORIES_COLS_ARR].astype("category")
            return concat_df
        else:
            return seasons_df_arr


def get_distance(df):
    df["home_shot_distance"] = df['HOMEDESCRIPTION'].str.extract("(\d{1,2}\')")
    df["visitor_shot_distance"] = df['VISITORDESCRIPTION'].str.extract("(\d{1,2}\')")

    df['home_shot_distance'] = df['home_shot_distance'].str.replace('\'', '').astype(float)
    # df['home_shot_distance'] = df['home_shot_distance'].fillna(-1).astype(int)
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'HOME_PLAYER') & (
        df['home_shot_distance'].isna()) & (df["HOMEDESCRIPTION"].str.contains("3PT ")), 'home_shot_distance'] = 23
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'HOME_PLAYER') & (
        df['home_shot_distance'].isna()), 'home_shot_distance'] = 0

    df['visitor_shot_distance'] = df['visitor_shot_distance'].str.replace('\'', '').astype(float)
    # df['visitor_shot_distance'] = df['visitor_shot_distance'].fillna(-1).astype(int)
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
        df['visitor_shot_distance'].isna()) & (
               df["VISITORDESCRIPTION"].str.contains("3PT ")), 'visitor_shot_distance'] = 23
    df.loc[(df["EVENTMSGTYPE"].str.startswith("FIELD_GOAL")) & (df["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
        df['visitor_shot_distance'].isna()), 'visitor_shot_distance'] = 0


# DEPRECATED TODO remove
# def load_game_data(columns=None, seasons=None, path=RAW_DATA_PATH, force_recompute=True):
#     GAME_ID_STR = "GAME_ID"
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     games_data_file = os.path.join(dir_path, "./../data/processed/load_data_games_arr_v1.pkl")
#
#     # reading from file or recomputing depending on the flags
#     if not force_recompute and os.path.exists(games_data_file):
#         print(f"Loading data from file {games_data_file} ...")
#         with open(games_data_file, "rb") as file:
#             game_data_df = pickle.load(file)
#         print("Data loaded!")
#         return game_data_df
#     else:
#         pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False)
#         print("Loaded PBP-data")
#         game_data_per_season = []
#         for pbp_data in pbp_data_per_season:
#             pbp_grouped = pbp_data.groupby(GAME_ID_STR)
#
#             # Create DataFrame with first column "play_count"
#             game_data = pbp_grouped.size().to_frame(name="play_count")
#             # Add arbitrarily many features:
#
#             # Season name is the same for all plays in a game: Just get first.
#             game_data["season_name"] = pbp_grouped["season_name"].first()
#             # Get date and start/end time would be nice.
#
#             # Visitor team:
#             game_data["visitor_team_id"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
#             game_data["visitor_team_city"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
#             game_data["visitor_team_nickname"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'VISITOR_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])
#             game_data["visitor_record_wins"] = 0
#             game_data["visitor_record_losses"] = 0
#             # Score:
#             game_data[["visitor_final_score", "home_final_score"]] = pbp_grouped.apply(
#                 lambda x: x[~x["SCORE"].isna()]["SCORE"].str.split(" - ", expand=True).astype(
#                     int).max())  # Complicated because in ca. 5 games the score column is messed up and out of order.
#             game_data["home_win"] = game_data["visitor_final_score"] < game_data["home_final_score"]
#             # Home team:
#             game_data["home_team_id"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_ID"].iloc[0])
#             game_data["home_team_city"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_CITY"].iloc[0])
#             game_data["home_team_nickname"] = pbp_grouped.apply(
#                 lambda x: x[x["PERSON1TYPE"] == 'HOME_PLAYER']["PLAYER1_TEAM_NICKNAME"].iloc[0])
#             # Calculated later
#             game_data["home_record_wins"] = 0
#             game_data["home_record_losses"] = 0
#
#             # Number of periods played: >4 is overtime
#             game_data["periods"] = pbp_grouped["PERIOD"].max()
#             # Minutes played:
#             game_data["minutes_played"] = 48 + (game_data["periods"] - 4) * 2.5
#
#             # Players deployed
#             game_data["visitor_players_deployed"] = pbp_grouped.apply(lambda x: len(set.union(
#                 *[set(x[x[f"PERSON{i}TYPE"] == "VISITOR_PLAYER"][f"PLAYER{i}_ID"].unique()) for i in range(1, 4)])))
#             game_data["home_players_deployed"] = pbp_grouped.apply(lambda x: len(
#                 set.union(
#                     *[set(x[x[f"PERSON{i}TYPE"] == "HOME_PLAYER"][f"PLAYER{i}_ID"].unique()) for i in range(1, 4)])))
#
#             # Field goal stats
#             game_data["visitor_fg_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["visitor_fg_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["visitor_3PT_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
#                     x["VISITORDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
#             game_data["visitor_3PT_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
#                     x["VISITORDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
#             game_data["home_fg_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_fg_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_3PT_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (
#                     x["HOMEDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
#             game_data["home_3PT_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FIELD_GOAL_MISSED') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (
#                     x["HOMEDESCRIPTION"].str.contains("3PT"))).groupby(level=0).sum()
#
#             # Free throw stats
#             game_data["visitor_ft_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
#                         x["VISITORDESCRIPTION"].str.contains("MISS") == False)).groupby(level=0).sum()
#             game_data["visitor_ft_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
#                     x["VISITORDESCRIPTION"].str.contains("MISS"))).groupby(level=0).sum()
#             game_data["home_ft_made"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (
#                         x["HOMEDESCRIPTION"].str.contains("MISS") == False)).groupby(level=0).sum()
#             game_data["home_ft_missed"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FREE_THROW') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (
#                     x["HOMEDESCRIPTION"].str.contains("MISS"))).groupby(level=0).sum()
#
#             # Rebound stats:
#             # Player rebounds the ball in live game
#             game_data["visitor_rebound"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER') & (
#                         x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
#             game_data["home_rebound"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_PLAYER') & (
#                         x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
#             # Team gets the ball if out of bounds
#             game_data["visitor_team_rebound"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'VISITOR_TEAM') & (
#                         x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
#             game_data["home_team_rebound"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'REBOUND') & (x["PERSON1TYPE"] == 'HOME_TEAM') & (
#                         x["EVENTMSGACTIONTYPE"] == "live")).groupby(level=0).sum()
#
#             # Turnover stats:
#             # Player turns the ball over: bad pass, offensive foul, ...
#             game_data["visitor_turnover"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_turnover"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#             # Team turn over: shot clock violation, 5 sec violation
#             game_data["visitor_team_turnover"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).groupby(
#                 level=0).sum()
#             game_data["home_team_turnover"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TURNOVER') & (x["PERSON1TYPE"] == 'HOME_TEAM')).groupby(level=0).sum()
#
#             # Foul stats:
#             game_data["visitor_foul"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(level=0).sum()
#             game_data["home_foul"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'FOUL') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(level=0).sum()
#
#             # Substitution stats:
#             game_data["visitor_subs"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_subs"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'SUBSTITUTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#
#             # Timeout stats:
#             game_data["visitor_timeout"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'VISITOR_TEAM')).groupby(
#                 level=0).sum()
#             game_data["home_timeout"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'TIMEOUT') & (x["PERSON1TYPE"] == 'HOME_TEAM')).groupby(level=0).sum()
#
#             # Jump ball stats:
#             game_data["visitor_jump_balls_won"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_jump_balls_won"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'JUMP_BALL') & (x["PERSON3TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["tip_off_winner"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'JUMP_BALL']["PERSON3TYPE"].iloc[0] if 'JUMP_BALL' in x[
#                     "EVENTMSGTYPE"].unique() else "UNKNOWN")
#
#             # Ejection stats:
#             # Player on the court gets ejected
#             game_data["visitor_ejection"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_PLAYER')).groupby(
#                 level=0).sum()
#             game_data["home_ejection"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_PLAYER')).groupby(
#                 level=0).sum()
#             # Non player gets ejected: coach etc.
#             game_data["visitor_team_ejection"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'VISITOR_TEAM_FOUL')).groupby(
#                 level=0).sum()
#             game_data["home_team_ejection"] = pbp_grouped.apply(
#                 lambda x: (x["EVENTMSGTYPE"] == 'EJECTION') & (x["PERSON1TYPE"] == 'HOME_TEAM_FOUL')).groupby(
#                 level=0).sum()
#
#             # Player performance:
#             # Points of scoring leader
#             game_data["home_scoring_leader"] = pbp_grouped.apply(lambda x: (
#                 (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["HOMEDESCRIPTION"].str.contains('3PT')) & (
#                         x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
#                     x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
#                             (x["HOMEDESCRIPTION"].str.contains('3PT')) == False) & (
#                               x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 2, fill_value=0).add(
#                     x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["HOMEDESCRIPTION"].str.contains('MISS')) == False) & (
#                             x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
#             ).idxmax())
#             game_data["home_scoring_leader_points"] = pbp_grouped.apply(lambda x: (
#                 (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["HOMEDESCRIPTION"].str.contains('3PT')) & (
#                         x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
#                     x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
#                             (x["HOMEDESCRIPTION"].str.contains('3PT')) == False) & (
#                               x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size() * 2, fill_value=0).add(
#                     x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & ((x["HOMEDESCRIPTION"].str.contains('MISS')) == False) & (
#                             x["PERSON1TYPE"] == 'HOME_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
#             ).max())
#             game_data["visitor_scoring_leader"] = pbp_grouped.apply(lambda x: (
#                 (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["VISITORDESCRIPTION"].str.contains('3PT')) & (
#                         x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
#                     x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
#                             (x["VISITORDESCRIPTION"].str.contains('3PT')) == False) & (
#                               x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 2,
#                     fill_value=0).add(
#                     x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & (
#                             (x["VISITORDESCRIPTION"].str.contains('MISS')) == False) & (
#                               x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
#             ).idxmax())
#             game_data["visitor_scoring_leader_points"] = pbp_grouped.apply(lambda x: (
#                 (x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (x["VISITORDESCRIPTION"].str.contains('3PT')) & (
#                         x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 3).add(
#                     x[(x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE') & (
#                             (x["VISITORDESCRIPTION"].str.contains('3PT')) == False) & (
#                               x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size() * 2,
#                     fill_value=0).add(
#                     x[(x["EVENTMSGTYPE"] == 'FREE_THROW') & (
#                             (x["VISITORDESCRIPTION"].str.contains('MISS')) == False) & (
#                               x["PERSON1TYPE"] == 'VISITOR_PLAYER')].groupby("PLAYER1_ID").size(), fill_value=0)
#             ).max())
#
#             # Shooting Distance information
#             # Max and min distance
#             game_data["home_made_max_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].max())
#             game_data["visitor_made_max_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].max())
#
#             game_data["home_made_min_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].min())
#             game_data["visitor_made_min_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].min())
#
#             game_data["home_made_mean_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["home_shot_distance"].mean())
#             game_data["visitor_made_mean_shot_distance"] = pbp_grouped.apply(
#                 lambda x: x[x["EVENTMSGTYPE"] == 'FIELD_GOAL_MADE']["visitor_shot_distance"].mean())
#
#             # Calculate record after game for both teams (total number of wins and losses for the season):
#             wins_dict = {team: 0 for team in
#                          set(game_data["visitor_team_id"].to_list() + game_data["home_team_id"].to_list())}
#             losses_dict = wins_dict.copy()
#             for i, row in game_data.iterrows():
#                 if row["home_win"]:
#                     wins_dict[row["home_team_id"]] += 1
#                     losses_dict[row["visitor_team_id"]] += 1
#                 else:
#                     wins_dict[row["visitor_team_id"]] += 1
#                     losses_dict[row["home_team_id"]] += 1
#                 game_data.at[i, "home_record_wins"] = wins_dict[row["home_team_id"]]
#                 game_data.at[i, "home_record_losses"] = losses_dict[row["home_team_id"]]
#                 game_data.at[i, "visitor_record_wins"] = wins_dict[row["visitor_team_id"]]
#                 game_data.at[i, "visitor_record_losses"] = losses_dict[row["visitor_team_id"]]
#
#             game_data_per_season.append(game_data)
#             print(f"Calculated game data for the {pbp_data['season_name'][0]} season", end="\r")
#
#         if game_data_per_season is None:
#             raise Exception("Game data is non-existent! Check for bugs")
#         else:
#             game_data_per_season = pd.concat(game_data_per_season)
#             # saving seasons arr to file, can be recomputed as single df
#             print("Saving to file ...")
#             with open(games_data_file, "wb") as file:
#                 pickle.dump(game_data_per_season, file)
#             print("Saved dataframe as file!")
#             return game_data_per_season


def load_game_data_zan(columns=None, seasons=None, path=RAW_DATA_PATH, force_recompute=True,
                       out_f_name="./../data/processed/load_data_games_arr_v2_zan.pkl"):
    """
    Method loops through all the seasons and games and calculates per game metrics.
    TODO finish documentation

    :param columns:
    :param seasons:
    :param path:
    :param force_recompute:
    :param out_f_name:
    :return:
    """
    GAME_ID_STR = "GAME_ID"
    data_columns = ["play_count", "home_team_id", "visitor_team_id", "home_record_wins", "home_record_losses"]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    games_data_file = os.path.join(dir_path, out_f_name)

    # reading from file or recomputing depending on the flags
    if not force_recompute and os.path.exists(games_data_file):
        print(f"Loading data from file {games_data_file} ...")
        with open(games_data_file, "rb") as file:
            game_data_df = pickle.load(file)
        print("Data loaded!")
        return game_data_df
    else:
        pbp_data_per_season = load_data(seasons=seasons, path=path, single_df=False, force_recompute=False)
        print("Loaded PBP-data")

        # initialized columns that don't need to be added as it goes
        games_df = pd.DataFrame(columns=data_columns)
        games_df.index.name = GAME_ID_STR
        for pbp_data in pbp_data_per_season:
            # Dictionaries for tracking the number of wins and losses for each team per season
            wins_dict = defaultdict(lambda: 0)
            losses_dict = defaultdict(lambda: 0)

            for game_id in pbp_data[GAME_ID_STR].sort_values().unique():
                # if there are any games with the same ID trigger exception
                if game_id in games_df.index:
                    raise Exception("Duplicate game index in games_df", game_id)
                pbp_grouped = pbp_data[pbp_data[GAME_ID_STR] == game_id]

                # Create DataFrame with first column "play_count"
                games_df.at[game_id, "play_count"] = len(pbp_grouped.index)
                # Add arbitrarily many features:

                # Season name is the same for all plays in a game: Just get first.
                games_df.at[game_id, "season_name"] = pbp_grouped["season_name"].iloc[0]
                # Get date and start/end time would be nice.

                # Visitor team:
                person1_visitor_mask = pbp_grouped["PERSON1TYPE"] == "VISITOR_PLAYER"
                visit_player_df = pbp_grouped[person1_visitor_mask]
                games_df.at[game_id, "visitor_team_id"] = visit_player_df["PLAYER1_TEAM_ID"].iloc[0].astype(int)
                games_df.at[game_id, "visitor_team_city"] = visit_player_df["PLAYER1_TEAM_CITY"].iloc[0]
                games_df.at[game_id, "visitor_team_nickname"] = visit_player_df["PLAYER1_TEAM_NICKNAME"].iloc[0]

                # Score:
                # Complicated because in ca. 5 games the score column is messed up and out of order.
                final_score = pbp_grouped[~pbp_grouped["SCORE"].isna()]["SCORE"].str.split(" - ", expand=True).astype(
                    int).max()

                games_df.at[game_id, "home_final_score"] = final_score[1]
                games_df.at[game_id, "visitor_final_score"] = final_score[0]
                games_df.at[game_id, "home_win"] = games_df.at[game_id, "visitor_final_score"] < games_df.at[
                    game_id, "home_final_score"]

                # Home team:
                person1_home_mask = (pbp_grouped["PERSON1TYPE"] == "HOME_PLAYER")
                home_player_df = pbp_grouped[person1_home_mask]
                games_df.at[game_id, "home_team_id"] = home_player_df["PLAYER1_TEAM_ID"].iloc[0].astype(int)
                games_df.at[game_id, "home_team_city"] = home_player_df["PLAYER1_TEAM_CITY"].iloc[0]
                games_df.at[game_id, "home_team_nickname"] = home_player_df["PLAYER1_TEAM_NICKNAME"].iloc[0]

                # Number of periods played: >4 is overtime
                games_df.at[game_id, "periods"] = pbp_grouped["PERIOD"].max()
                # Minutes played:
                games_df.at[game_id, "minutes_played"] = 48 + (games_df.at[game_id, "periods"] - 4) * 2.5

                # Players deployed
                vis_players_deployed_arr = [set(pbp_grouped[person1_visitor_mask]["PLAYER1_ID"].unique())\
                    .union(pbp_grouped[pbp_grouped["PERSON2TYPE"] == "VISITOR_PLAYER"]["PLAYER2_ID"].unique())\
                    .union(pbp_grouped[pbp_grouped["PERSON3TYPE"] == "VISITOR_PLAYER"]["PLAYER3_ID"].unique())]

                games_df.at[game_id, "visitor_players_deployed_ids"] = vis_players_deployed_arr
                games_df.at[game_id, "visitor_players_deployed"] = len(vis_players_deployed_arr[0])

                home_players_deployed_arr = [set(pbp_grouped[person1_home_mask]["PLAYER1_ID"].unique())\
                        .union(pbp_grouped[pbp_grouped["PERSON2TYPE"] == "HOME_PLAYER"]["PLAYER2_ID"].unique())\
                        .union(pbp_grouped[pbp_grouped["PERSON3TYPE"] == "HOME_PLAYER"]["PLAYER3_ID"].unique())]
                games_df.at[game_id, "home_players_deployed_ids"] = home_players_deployed_arr
                games_df.at[game_id, "home_players_deployed"] = len(home_players_deployed_arr[0])

                # Used masks
                f_goal_made_m = pbp_grouped["EVENTMSGTYPE"] == "FIELD_GOAL_MADE"
                f_goal_missed_m = pbp_grouped["EVENTMSGTYPE"] == "FIELD_GOAL_MISSED"

                # Field goal stats
                games_df.at[game_id, "visitor_fg_made"] = len(pbp_grouped[f_goal_made_m & person1_visitor_mask].index)
                games_df.at[game_id, "visitor_fg_missed"] = len(
                    pbp_grouped[f_goal_missed_m & person1_visitor_mask].index)
                games_df.at[game_id, "visitor_3PT_made"] = len(pbp_grouped[f_goal_made_m & person1_visitor_mask
                                                                           & (pbp_grouped[
                    "VISITORDESCRIPTION"].str.contains(
                    "3PT"))].index)
                games_df.at[game_id, "visitor_3PT_missed"] = len(pbp_grouped[f_goal_missed_m
                                                                             & person1_visitor_mask
                                                                             & (pbp_grouped[
                    "VISITORDESCRIPTION"].str.contains(
                    "3PT"))].index)

                # home
                games_df.at[game_id, "home_fg_made"] = len(pbp_grouped[f_goal_made_m & person1_home_mask].index)
                games_df.at[game_id, "home_fg_missed"] = len(pbp_grouped[f_goal_missed_m & person1_home_mask].index)
                games_df.at[game_id, "home_3PT_made"] = len(pbp_grouped[f_goal_made_m & person1_home_mask
                                                                        & (pbp_grouped["HOMEDESCRIPTION"].str.contains(
                    "3PT"))].index)
                games_df.at[game_id, "home_3PT_missed"] = len(pbp_grouped[f_goal_missed_m
                                                                          & person1_home_mask & (pbp_grouped[
                    "HOMEDESCRIPTION"].str.contains(
                    "3PT"))].index)

                # Free throw stats
                games_df.at[game_id, "visitor_ft_made"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW")
                                                                          & person1_visitor_mask
                                                                          & (pbp_grouped[
                                                                                 "VISITORDESCRIPTION"].str.contains(
                    "MISS") == False)].index)
                games_df.at[game_id, "visitor_ft_missed"] = len(
                    pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW")
                                & person1_visitor_mask
                                & (pbp_grouped["VISITORDESCRIPTION"].str.contains("MISS"))].index)
                games_df.at[game_id, "home_ft_made"] = len(
                    pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW") & person1_home_mask
                                & (pbp_grouped["HOMEDESCRIPTION"].str.contains("MISS") == False)].index)
                games_df.at[game_id, "home_ft_missed"] = len(
                    pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW") & person1_home_mask
                                & (pbp_grouped["HOMEDESCRIPTION"].str.contains("MISS"))].index)

                # Rebound stats:
                # Player rebounds the ball in live game
                rebound_mask = pbp_grouped["EVENTMSGTYPE"] == "REBOUND"
                games_df.at[game_id, "visitor_rebound"] = len(pbp_grouped[rebound_mask & person1_visitor_mask
                                                                          & (pbp_grouped[
                                                                                 "EVENTMSGACTIONTYPE"] == "live")].index)
                games_df.at[game_id, "home_rebound"] = len(
                    pbp_grouped[rebound_mask & person1_home_mask & (pbp_grouped["EVENTMSGACTIONTYPE"] == "live")].index)

                # Team gets the ball if out of bounds
                games_df.at[game_id, "visitor_team_rebound"] = len(pbp_grouped[rebound_mask
                                                                               & (pbp_grouped["PERSON1TYPE"] == "VISITOR_TEAM")
                                                                               & (pbp_grouped[
                                                                                      "EVENTMSGACTIONTYPE"] == "live")].index)

                games_df.at[game_id, "home_team_rebound"] = len(
                    pbp_grouped[rebound_mask & (pbp_grouped["PERSON1TYPE"] == 'HOME_TEAM')
                                & (pbp_grouped["EVENTMSGACTIONTYPE"] == "live")].index)

                # Turnover stats:
                # Player turns the ball over: bad pass, offensive foul, ...
                turnover_mask = pbp_grouped["EVENTMSGTYPE"] == "TURNOVER"
                games_df.at[game_id, "visitor_turnover"] = len(pbp_grouped[turnover_mask & person1_visitor_mask].index)
                games_df.at[game_id, "home_turnover"] = len(pbp_grouped[turnover_mask & person1_home_mask].index)

                # Team turn over: shot clock violation, 5 sec violation
                games_df.at[game_id, "visitor_team_turnover"] = len(
                    pbp_grouped[turnover_mask & (pbp_grouped["PERSON1TYPE"] == 'VISITOR_TEAM')].index)
                games_df.at[game_id, "home_team_turnover"] = len(
                    pbp_grouped[turnover_mask & (pbp_grouped["PERSON1TYPE"] == 'HOME_TEAM')].index)

                # Foul stats:
                games_df.at[game_id, "visitor_foul"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FOUL")
                                                                       & person1_visitor_mask].index)
                games_df.at[game_id, "home_foul"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FOUL")
                                                                    & person1_home_mask].index)
                # Substitution stats:
                games_df.at[game_id, "visitor_subs"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "SUBSTITUTION")
                                                                       & person1_visitor_mask].index)
                games_df.at[game_id, "home_subs"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "SUBSTITUTION")
                                                                    & person1_home_mask].index)

                # Timeout stats:
                games_df.at[game_id, "visitor_timeout"] = len(pbp_grouped[
                                                                  (pbp_grouped["EVENTMSGTYPE"] == "TIMEOUT") & (
                                                                          pbp_grouped[
                                                                              "PERSON1TYPE"] == 'VISITOR_TEAM')].index)
                games_df.at[game_id, "home_timeout"] = len(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "TIMEOUT") & (
                        pbp_grouped["PERSON1TYPE"] == 'HOME_TEAM')].index)

                # Jump ball stats:
                games_df.at[game_id, "visitor_jump_balls_won"] = len(
                    pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "JUMP_BALL")
                                & (pbp_grouped["PERSON3TYPE"] == 'VISITOR_PLAYER')].index)
                games_df.at[game_id, "home_jump_balls_won"] = len(pbp_grouped[
                                                                      (pbp_grouped["EVENTMSGTYPE"] == "JUMP_BALL") & (
                                                                              pbp_grouped[
                                                                                  "PERSON3TYPE"] == 'HOME_PLAYER')].index)


                games_df.at[game_id, "tip_off_winner"] = \
                    pbp_grouped[pbp_grouped["EVENTMSGTYPE"] == "JUMP_BALL"]["PERSON3TYPE"].iloc[0] if 'JUMP_BALL' in \
                                                                                                      pbp_grouped[
                                                                                                          "EVENTMSGTYPE"].unique() else "UNKNOWN"

                # Ejection stats:
                # Player on the court gets ejected
                eject_mask = pbp_grouped["EVENTMSGTYPE"] == "EJECTION"
                games_df.at[game_id, "visitor_ejection"] = len(pbp_grouped[eject_mask & person1_visitor_mask].index)
                games_df.at[game_id, "home_ejection"] = len(pbp_grouped[eject_mask & person1_home_mask].index)

                # Non player gets ejected: coach etc.
                games_df.at[game_id, "visitor_team_ejection"] = len(
                    pbp_grouped[eject_mask & (pbp_grouped["PERSON1TYPE"] == 'VISITOR_TEAM_FOUL')].index)
                games_df.at[game_id, "home_team_ejection"] = len(
                    pbp_grouped[eject_mask & (pbp_grouped["PERSON1TYPE"] == 'HOME_TEAM_FOUL')].index)

                # Player performance:
                # Points of scoring leader
                home_scoring_l = ((pbp_grouped[(f_goal_made_m) & (pbp_grouped["HOMEDESCRIPTION"].str.contains('3PT'))
                                               & person1_home_mask].groupby("PLAYER1_ID").size() * 3)
                                  .add(pbp_grouped[(f_goal_made_m)
                                                   & ((pbp_grouped["HOMEDESCRIPTION"].str.contains('3PT')) == False)
                                                   & person1_home_mask].groupby("PLAYER1_ID").size() * 2, fill_value=0)
                                  .add(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW")
                                                   & ((pbp_grouped["HOMEDESCRIPTION"].str.contains('MISS')) == False)
                                                   & person1_home_mask].groupby("PLAYER1_ID").size(), fill_value=0))

                games_df.at[game_id, "home_scoring_leader"] = home_scoring_l.idxmax()
                games_df.at[game_id, "home_scoring_leader_points"] = home_scoring_l.max()

                visitor_scoring_l = ((pbp_grouped[(f_goal_made_m)
                                                  & (pbp_grouped["VISITORDESCRIPTION"].str.contains('3PT'))
                                                  & person1_visitor_mask].groupby("PLAYER1_ID").size() * 3)
                                     .add(pbp_grouped[(f_goal_made_m)
                                                      & ((pbp_grouped["VISITORDESCRIPTION"].str.contains(
                    '3PT')) == False)
                                                      & person1_visitor_mask].groupby("PLAYER1_ID").size() * 2,
                                          fill_value=0)
                                     .add(pbp_grouped[(pbp_grouped["EVENTMSGTYPE"] == "FREE_THROW")
                                                      & ((pbp_grouped["VISITORDESCRIPTION"].str.contains(
                    'MISS')) == False)
                                                      & person1_visitor_mask].groupby("PLAYER1_ID").size(),
                                          fill_value=0))
                games_df.at[game_id, "visitor_scoring_leader"] = visitor_scoring_l.idxmax()
                games_df.at[game_id, "visitor_scoring_leader_points"] = visitor_scoring_l.max()

                # Shooting Distance information
                # Max and min distance
                games_df.at[game_id, "home_made_max_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "home_shot_distance"].max()
                games_df.at[game_id, "visitor_made_max_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "visitor_shot_distance"].max()
                games_df.at[game_id, "home_made_min_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "home_shot_distance"].min()
                games_df.at[game_id, "visitor_made_min_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "visitor_shot_distance"].min()

                games_df.at[game_id, "home_made_mean_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "home_shot_distance"].mean()
                games_df.at[game_id, "visitor_made_mean_shot_distance"] = pbp_grouped[f_goal_made_m][
                    "visitor_shot_distance"].mean()

                # Calculate record after game for both teams (total number of wins and losses for the season):
                if games_df.at[game_id, "home_win"]:
                    wins_dict[games_df.at[game_id, "home_team_id"]] += 1
                    losses_dict[games_df.at[game_id, "visitor_team_id"]] += 1
                else:
                    wins_dict[games_df.at[game_id, "visitor_team_id"]] += 1
                    losses_dict[games_df.at[game_id, "home_team_id"]] += 1
                games_df.at[game_id, "home_record_wins"] = wins_dict[games_df.at[game_id, "home_team_id"]]
                games_df.at[game_id, "home_record_losses"] = losses_dict[games_df.at[game_id, "home_team_id"]]
                games_df.at[game_id, "visitor_record_wins"] = wins_dict[games_df.at[game_id, "visitor_team_id"]]
                games_df.at[game_id, "visitor_record_losses"] = losses_dict[games_df.at[game_id, "visitor_team_id"]]

            print(f"Calculated game data for the {pbp_data['season_name'][0]} season")

        # extra feature for combined games, which can be done after all games are processed
        games_df["games_already_played_in_season"] = games_df["visitor_record_wins"] + games_df["visitor_record_losses"]

        # true shooting percentages
        games_df["home_TSP"] = games_df["home_final_score"] / (2 * (
                (games_df["home_fg_made"] + games_df["home_fg_missed"]) +
                (0.44 * (games_df["home_3PT_made"] + games_df["home_3PT_missed"]))))
        games_df["visitor_TSP"] = games_df["visitor_final_score"] / (2 * (
                (games_df["visitor_fg_made"] + games_df["visitor_fg_missed"]) +
                (0.44 * (games_df["visitor_3PT_made"] + games_df["visitor_3PT_missed"]))))

        # game point difference
        games_df["home_final_score_diff"] = games_df["home_final_score"] - games_df["visitor_final_score"]
        games_df["visitor_final_score_diff"] = games_df["visitor_final_score"] - games_df["home_final_score"]

        # most common lineups
        print("Concatenating common lineups")
        common_lineups = pd.read_pickle("../../data/processed/common_lineups.pkl")
        games_df = pd.concat([games_df, common_lineups], axis=1)

        if len(games_df.index) < 1:
            raise Exception("Game data is non-existent! Check for bugs")
        else:
            int_columns = ["play_count", "home_team_id", "visitor_team_id", "home_win", "home_final_score",
                           "home_record_wins", "home_record_losses", "visitor_final_score", "visitor_record_wins",
                           "periods", "minutes_played", "visitor_players_deployed", "home_players_deployed",
                           "visitor_fg_made", "visitor_fg_missed", "visitor_3PT_made",
                           "visitor_3PT_missed", "home_fg_made", "home_fg_missed", "home_3PT_made",
                           "home_3PT_missed", "visitor_ft_made", "visitor_ft_missed",
                           "home_ft_made", "home_ft_missed", "visitor_rebound", "home_rebound",
                           "visitor_team_rebound", "home_team_rebound", "visitor_turnover",
                           "home_turnover", "visitor_team_turnover", "home_team_turnover",
                           "visitor_foul", "home_foul", "visitor_subs", "home_subs",
                           "visitor_timeout", "home_timeout", "visitor_jump_balls_won",
                           "home_jump_balls_won", "visitor_ejection",
                           "home_ejection", "visitor_team_ejection", "home_team_ejection",
                           "home_scoring_leader", "home_scoring_leader_points",
                           "visitor_scoring_leader", "visitor_scoring_leader_points",
                           "home_made_max_shot_distance", "visitor_made_max_shot_distance",
                           "home_made_min_shot_distance", "visitor_made_min_shot_distance",
                           "visitor_record_losses", "games_already_played_in_season", 
                           "home_common_lineup", "visitor_common_lineup"]

            games_df[int_columns] = games_df[int_columns].astype(int)
            # saving seasons arr to file, can be recomputed as single df
            print("Saving to file ...")
            with open(games_data_file, "wb") as file:
                pickle.dump(games_df, file)
            print("Saved dataframe as file!")
            return games_df


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


def player_data_preprocessing():
    """
    Preprocess player data for further analysis.
    :return:
    """
    # player_info = pd.read_csv("./../../data/raw/player-data/player_info.csv")
    # lineups = pd.read_csv("./../../data/processed/lineups-all-seasons.csv")
    # temp_lineups = deepcopy(lineups_df)
    player_df = load_player_data()
    player_df = player_data_convert_to_metric_units(player_df)
    # lineups = load_lineups()
    # player_df = add_extra_player_features_gregor(player_df, lineups)

    return player_df


def player_data_convert_to_metric_units(player_df, remove_nan_rows=False):
    """
    Converts  mainly it converts from pounds to kg and from feet/inches to cm.

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


def add_extra_player_features_gregor(player_df, lineups_df):
    """
    Adds extra features to the player data. (TODO @gregor add extra info)

    :param player_df:
    :param lineups_df:
    :return:
    """
    temp_lineups = pd.DataFrame()
    temp_player_df = deepcopy(player_df)

    # Add season string to every lineup row
    temp_lineups["season"] = lineups_df["game_id"].astype(str).str[1:3].apply(lambda x: f"20{x}-20{int(x) + 1:02d}")

    # This can be optimized if needed
    temp_player_df["games_started"] = temp_player_df.apply(lambda x:
                                                           (temp_lineups[(temp_lineups["season"] == x["Season"]) & (
                                                                   (temp_lineups["home_player1_id"] == x["Player ID"]) |
                                                                   (temp_lineups["home_player2_id"] == x["Player ID"]) |
                                                                   (temp_lineups["home_player3_id"] == x["Player ID"]) |
                                                                   (temp_lineups["home_player4_id"] == x["Player ID"]) |
                                                                   (temp_lineups["home_player5_id"] == x["Player ID"]) |
                                                                   (temp_lineups["visitor_player1_id"] == x[
                                                                       "Player ID"]) |
                                                                   (temp_lineups["visitor_player2_id"] == x[
                                                                       "Player ID"]) |
                                                                   (temp_lineups["visitor_player3_id"] == x[
                                                                       "Player ID"]) |
                                                                   (temp_lineups["visitor_player4_id"] == x[
                                                                       "Player ID"]) |
                                                                   (temp_lineups["visitor_player5_id"] == x[
                                                                       "Player ID"]))]["game_id"].count()), axis=1)
    return temp_player_df


def load_lineups():
    # TODO @gregor do this
    pass

def add_recent_stats(game_data, recent_range=5):
    game_data = game_data.copy().sort_index().reset_index()
    unique_team_ids = set(game_data["visitor_team_id"].to_list())
    games_by_team_id = {team_id:game_data[(game_data["visitor_team_id"]==team_id)|(game_data["home_team_id"]==team_id)] for team_id in unique_team_ids}
    
    for team in ["home","visitor"]:
        game_data[f"{team}_recent_home_game_ratio"] = 0
        game_data[f"{team}_recent_win_ratio"] = 0
        game_data[f"{team}_recent_points"] = 0
        for shot in ["fg","3PT","ft"]:
            for result in ["made","missed"]:
                game_data[f"{team}_recent_{shot}_{result}"] = 0
        for feature in ["players_deployed","rebound","turnover","foul"]:
            game_data[f"{team}_recent_{feature}"] = 0
    
    for i,game in game_data.iterrows():
        for team in ["home","visitor"]:
            team_id = game[f"{team}_team_id"]
            recent_games = games_by_team_id[team_id].loc[:i-1].tail(recent_range)
            recent_window = len(recent_games)
    
            game_data.at[i, f"{team}_recent_home_game_ratio"] = len(recent_games[recent_games["home_team_id"]==team_id]) / recent_window if recent_window>0 else 0
            game_data.at[i, f"{team}_recent_win_ratio"] = len(recent_games[((recent_games["visitor_team_id"]==team_id)&(recent_games["home_win"]==False))|((recent_games["home_team_id"]==team_id)&(recent_games["home_win"]==True))]) / recent_window if recent_window>0 else 0
            game_data.at[i, f"{team}_recent_points"] = (recent_games[recent_games["visitor_team_id"]==team_id]["visitor_final_score"].sum() + recent_games[recent_games["home_team_id"]==team_id]["home_final_score"].sum()) / recent_window if recent_window>0 else 0
            for shot in ["fg","3PT","ft"]:
                for result in ["made","missed"]:
                    game_data.at[i, f"{team}_recent_{shot}_{result}"] = (recent_games[recent_games["visitor_team_id"]==team_id][f"visitor_{shot}_{result}"].sum() + recent_games[recent_games["home_team_id"]==team_id][f"home_{shot}_{result}"].sum()) / recent_window if recent_window>0 else 0
            for feature in ["players_deployed","rebound","turnover","foul"]:
                    game_data.at[i, f"{team}_recent_{feature}"] = (recent_games[recent_games["visitor_team_id"]==team_id][f"visitor_{feature}"].sum() + recent_games[recent_games["home_team_id"]==team_id][f"home_{feature}"].sum()) / recent_window if recent_window>0 else 0
    return game_data

