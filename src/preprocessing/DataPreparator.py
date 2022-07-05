import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.data_loader import load_game_data_zan
from sklearn.preprocessing import LabelEncoder


class DataPreparator:

    def __init__(self, game_file_name="", player_file_name="./../../data/raw/player-data/player_info.csv",
                 force_recompute=False):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.game_file_name = os.path.join(dir_path, game_file_name)
        self.player_file_name = os.path.join(dir_path, player_file_name)
        self.force_recompute = force_recompute

        # used for encoding team names
        self.label_enc = LabelEncoder()

        self.games_df = self.load_game_df()
        self.games_df = self.add_recent_stats(self.games_df)
        self.players_df = self.load_player_df()

        self.general_data_dict = self.prepare_query_dict_structure()

        self.x_home_cols, self.x_visit_cols = None, None  # self.select_columns_for_predictions()
        # validation sets to be used at the end
        self.x_val = None
        self.y_val = None

    def add_recent_stats(self, game_data, recent_range=10):
        print("\nAdding recent stats to the dataframe...")
        game_data = game_data.copy().sort_index().reset_index()
        unique_team_ids = set(game_data["visitor_team_id"].to_list())
        games_by_team_id = {
            team_id: game_data[(game_data["visitor_team_id"] == team_id) | (game_data["home_team_id"] == team_id)] for
            team_id in unique_team_ids}

        for team in ["home", "visitor"]:
            game_data[f"{team}_recent_home_game_ratio"] = 0
            game_data[f"{team}_recent_win_ratio"] = 0
            game_data[f"{team}_recent_points"] = 0
            game_data[f"{team}_recent_TSP"] = 0

            for shot in ["fg", "3PT", "ft"]:
                for result in ["made", "missed"]:
                    game_data[f"{team}_recent_{shot}_{result}"] = 0
            for feature in ["players_deployed", "rebound", "turnover", "foul"]:
                game_data[f"{team}_recent_{feature}"] = 0

        for i, game in game_data.iterrows():
            for team in ["home", "visitor"]:
                team_id = game[f"{team}_team_id"]
                recent_games = games_by_team_id[team_id].loc[:i - 1].tail(recent_range)
                recent_window = len(recent_games) if len(recent_games) > 0 else np.NAN
                # could add to skip the lines if recent window is nan

                game_data.at[i, f"{team}_recent_home_game_ratio"] = len(recent_games[recent_games[
                                                                                         "home_team_id"] == team_id]) / recent_window
                game_data.at[i, f"{team}_recent_win_ratio"] = len(
                    recent_games[((recent_games["visitor_team_id"] == team_id)
                                  & (recent_games["home_win"] == False)) | (
                                         (recent_games[
                                              "home_team_id"] == team_id) & (
                                                 recent_games[
                                                     "home_win"] == True))]) / recent_window
                game_data.at[i, f"{team}_recent_points"] = (recent_games[recent_games["visitor_team_id"] == team_id][
                                                                "visitor_final_score"].sum() +
                                                            recent_games[recent_games["home_team_id"] == team_id][
                                                                "home_final_score"].sum()) / recent_window

                game_data.at[i, f"{team}_recent_TSP"] = (recent_games[recent_games["visitor_team_id"] == team_id][
                                                             "home_TSP"].sum() +
                                                         recent_games[recent_games["home_team_id"] == team_id][
                                                             "visitor_TSP"].sum()) / recent_window
                for shot in ["fg", "3PT", "ft"]:
                    for result in ["made", "missed"]:
                        game_data.at[i, f"{team}_recent_{shot}_{result}"] = (recent_games[recent_games[
                                                                                              "visitor_team_id"] == team_id][
                                                                                 f"visitor_{shot}_{result}"].sum() +
                                                                             recent_games[recent_games[
                                                                                              "home_team_id"] == team_id][
                                                                                 f"home_{shot}_{result}"].sum()) / recent_window
                for feature in ["players_deployed", "rebound", "turnover", "foul"]:
                    game_data.at[i, f"{team}_recent_{feature}"] = (recent_games[
                                                                       recent_games["visitor_team_id"] == team_id][
                                                                       f"visitor_{feature}"].sum() + recent_games[
                                                                       recent_games["home_team_id"] == team_id][
                                                                       f"home_{feature}"].sum()) / recent_window
        return game_data

    @staticmethod
    def select_columns_for_predictions(recent_cols=True):
        """
        TODO - add description
        :return:
        """
        if not recent_cols:
            x_common_cols = ["home_win", "periods", "minutes_played", "tip_off_winner",
                             "games_already_played_in_season"]
            x_home_cols = x_common_cols + ["home_team_id", "home_record_wins", "home_record_losses",
                                           "home_final_score", "home_fg_made", "home_fg_missed",
                                           "home_3PT_made", "home_3PT_missed", "home_ft_made", "home_ft_missed",
                                           "home_rebound", "home_team_rebound", "home_turnover",
                                           "home_team_turnover", "home_foul", "home_subs", "home_timeout",
                                           "home_jump_balls_won",
                                           "home_ejection", "home_team_ejection", "home_scoring_leader",
                                           "home_scoring_leader_points", "home_made_max_shot_distance",
                                           "home_made_min_shot_distance", "home_made_mean_shot_distance"]
            x_visit_cols = x_common_cols + ["visitor_team_id", "visitor_record_wins", "visitor_record_losses",
                                            "visitor_final_score", "visitor_fg_made", "visitor_fg_missed",
                                            "visitor_3PT_made", "visitor_3PT_missed", "visitor_ft_made",
                                            "visitor_ft_missed",
                                            "visitor_rebound", "visitor_team_rebound", "visitor_turnover",
                                            "visitor_team_turnover", "visitor_foul", "visitor_subs",
                                            "visitor_timeout",
                                            "visitor_jump_balls_won",
                                            "visitor_ejection", "visitor_team_ejection", "visitor_scoring_leader",
                                            "visitor_scoring_leader_points", "visitor_made_max_shot_distance",
                                            "visitor_made_min_shot_distance", "visitor_made_mean_shot_distance"]
            print(f"\nColumns that will be used are of lengths: \n"
                  f"    - x_home_cols: {len(x_home_cols)}, \n"
                  f"    - x_visit_cols length: {len(x_visit_cols)}")
        else:
            x_home_cols = ["home_team_id", "home_recent_TSP", "home_recent_home_game_ratio", "home_recent_win_ratio",
                           "home_recent_points", "home_recent_fg_made",
                           "home_recent_fg_missed", "home_recent_3PT_made",
                           "home_recent_3PT_missed", "home_recent_ft_made",
                           "home_recent_ft_missed", "home_recent_players_deployed",
                           "home_recent_rebound", "home_recent_turnover", "home_recent_foul"]

            x_visit_cols = ["visitor_team_id", "visitor_recent_TSP", "visitor_recent_home_game_ratio",
                            "visitor_recent_win_ratio",
                            "visitor_recent_points", "visitor_recent_fg_made",
                            "visitor_recent_fg_missed", "visitor_recent_3PT_made",
                            "visitor_recent_3PT_missed", "visitor_recent_ft_made",
                            "visitor_recent_ft_missed", "visitor_recent_players_deployed",
                            "visitor_recent_rebound", "visitor_recent_turnover", "visitor_recent_foul"]
            print(f"\nColumns that will be used are of lengths: \n"
                  f"    - x_home_cols: {len(x_home_cols)}, \n"
                  f"    - x_visit_cols length: {len(x_visit_cols)}")
        assert len(x_home_cols) == len(x_visit_cols), "x_home_cols and x_visit_cols should have the same length!"

        return x_home_cols, x_visit_cols

    @staticmethod
    def normalize_columns(games_df):
        # not necessarily the same as original columns, for example team id can't be normalized
        columns_to_scale = ["home_recent_TSP", "home_recent_home_game_ratio", "home_recent_win_ratio",
                            "home_recent_points", "home_recent_fg_made",
                            "home_recent_fg_missed", "home_recent_3PT_made",
                            "home_recent_3PT_missed", "home_recent_ft_made",
                            "home_recent_ft_missed", "home_recent_players_deployed",
                            "home_recent_rebound", "home_recent_turnover", "home_recent_foul"] \
                           + ["visitor_recent_TSP", "visitor_recent_home_game_ratio",
                              "visitor_recent_win_ratio",
                              "visitor_recent_points", "visitor_recent_fg_made",
                              "visitor_recent_fg_missed", "visitor_recent_3PT_made",
                              "visitor_recent_3PT_missed", "visitor_recent_ft_made",
                              "visitor_recent_ft_missed", "visitor_recent_players_deployed",
                              "visitor_recent_rebound", "visitor_recent_turnover", "visitor_recent_foul"]

        df_to_scale_arr = games_df[columns_to_scale].to_numpy()
        min_max_scaler = preprocessing.StandardScaler()
        scaled_columns = min_max_scaler.fit_transform(df_to_scale_arr)
        games_df[columns_to_scale] = scaled_columns

    # noinspection PyUnresolvedReferences
    def prepare_data_splits(self, season, data_split=None):
        if data_split is None:
            data_split = {"train": 0.8, "test": 0.1, "validation": 0.1}
        assert sum(data_split.values()) == 1, "Data split should sum to 1!"

        curr_season_df = self.get_games_df_for_season(season)
        # encoding string and ID variables to labels
        curr_season_df["tip_off_winner"] = (curr_season_df["tip_off_winner"] == "HOME_PLAYER").astype(int)
        curr_season_df["home_team_id"] = self.label_enc.fit_transform(curr_season_df["home_team_id"])
        curr_season_df["visitor_team_id"] = self.label_enc.fit_transform(curr_season_df["visitor_team_id"])
        curr_season_df["visitor_scoring_leader"] = self.label_enc.fit_transform(
            curr_season_df["visitor_scoring_leader"])
        curr_season_df["home_scoring_leader"] = self.label_enc.fit_transform(curr_season_df["home_scoring_leader"])

        x_matrix, y_vector = self.prepare_x_matrix_and_y_vector(curr_season_df)

        # splitting the data into train, test and validation
        data_len = len(x_matrix)
        train_len = int(data_len * data_split["train"])
        test_len = int(data_len * data_split["test"])
        val_len = int(data_len * data_split["validation"])

        # prepare the data for the train, test and validation
        x_train = x_matrix[:train_len]
        y_train = y_vector[:train_len]

        x_test = x_matrix[train_len:train_len + test_len]
        y_test = y_vector[train_len:train_len + test_len]

        self.x_val = x_matrix[train_len + test_len:]
        self.y_val = y_vector[train_len + test_len:]

        return x_train, y_train, x_test, y_test

    def prepare_x_matrix_and_y_vector(self, games_df, split_game_in_two_data_p=False, normalize_columns=True):
        self.x_home_cols, self.x_visit_cols = self.select_columns_for_predictions()

        assert len(self.x_home_cols) == len(self.x_visit_cols), "x_home_cols and x_visit_cols should have the same " \
                                                                "length! "
        if normalize_columns:
            self.normalize_columns(games_df)

        x_matrix_arr, y_vector_arr = None, None
        if split_game_in_two_data_p:
            # Two times as many rows/data points will be generated as there are games in the season
            # because each game will be represented by two rows in the matrix, win team stats and losing team stats
            x_matrix_arr = np.zeros((len(games_df.index) * 2, len(self.x_home_cols)))
            y_vector_arr = np.zeros((len(games_df.index) * 2))
            i_counter = 0
            for index, row in games_df.iterrows():
                # split data depending on who won the game, each game becomes two data points
                if row["home_win"] == 1:
                    win_team = row[self.x_home_cols].to_numpy()
                    lose_team = row[self.x_visit_cols].to_numpy()
                elif row["home_win"] == 0:
                    win_team = row[self.x_visit_cols].to_numpy()
                    lose_team = row[self.x_home_cols].to_numpy()
                else:
                    raise Exception("home_win should be either 0 or 1!")

                x_matrix_arr[i_counter, :] = win_team
                x_matrix_arr[i_counter + 1, :] = lose_team
                y_vector_arr[i_counter] = 1
                y_vector_arr[i_counter + 1] = 0
                i_counter += 2
        else:
            h_and_v_cols = self.x_home_cols + self.x_visit_cols
            x_matrix_arr = np.zeros((len(games_df.index), len(h_and_v_cols)))
            y_vector_arr = np.zeros((len(games_df.index)))
            i_counter = 0
            for index, row in games_df.iterrows():
                # split data depending on who won the game, each game becomes two data points
                data_row = row[h_and_v_cols].to_numpy()

                x_matrix_arr[i_counter, :] = data_row
                y_vector_arr[i_counter] = row["home_win"]
                i_counter += 1

        if x_matrix_arr is None or y_vector_arr is None:
            raise Exception("Unexpected error, x_matrix_arr or y_vector_arr is None!")

        return x_matrix_arr, y_vector_arr

    def prepare_query_dict_structure(self, output_file=None):
        """
         Function will prepare the dataset and all its features for the season based predictions.

         :param output_file: None by default, if not None, the prepared data will be saved to a file.
         :return: Dictionary with the prepared data:
         {
             "all_seasons_info": {
                 "team_ids_arr": <all team ids>
                 "player_ids_arr": <all player ids>
                 "games_played": <count of all games played>
                 ... here features which hold true for the whole dataset are added
             },
             "seasons": {
                 "2001-2002": {
                     "games": pd.Dataframe of all the games in that seasons with precomputed features
                     "player_info": pd.Dataframe of all the player data in that season (from file not from data loader)
                 },
                 ...
                 "2018-2019": {
                     "games": pd.Dataframe of all the games in that seasons with precomputed features
                     "player_info": pd.Dataframe of all the player data in that season (from file not from data loader)
                 },
         }
         Data could be also saved as JSON but the pandas structure would then have to be unraveled.

         """
        general_nba_info_dict = {
            "all_seasons_info": {
                "team_ids_arr": list(set(self.games_df["visitor_team_id"]).union(self.games_df["home_team_id"])),
                "player_ids_arr": list(self.players_df["Player ID"].unique()),
                "games_played": len(self.games_df.index),
            },
            "seasons":
                dict.fromkeys(list(self.games_df["season_name"].unique()), {
                    "games": None,
                    "player_info": None
                }),
        }

        for s_name in self.games_df["season_name"].unique():
            general_nba_info_dict["seasons"][str(s_name)]["games"] = self.games_df[
                self.games_df["season_name"] == s_name]
            general_nba_info_dict["seasons"][str(s_name)]["player_info"] = self.players_df[
                self.players_df["Season"] == s_name]

        if output_file is not None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # "./../../data/processed/prepared_predictions_dict.pkl"
            file_path = os.path.join(dir_path, output_file)

            with open(file_path, "wb") as f:
                pickle.dump(general_nba_info_dict, f)

        return general_nba_info_dict

    def get_games_df_for_season(self, season_name):
        """

        :param season_name: String of the season name which we want.
        :return: pd.DataFrame with all the games for the given season.
        """
        return deepcopy(self.general_data_dict["seasons"][season_name]["games"])

    def get_players_df_for_season(self, season_name):
        return deepcopy(self.general_data_dict["seasons"][season_name]["player_info"])

    def load_game_df(self):
        games_df = load_game_data_zan(force_recompute=self.force_recompute)
        games_df["season_name"] = games_df["season_name"].str.replace("-", "-20")

        return games_df

    def load_player_df(self):
        players_df = pd.read_csv(self.player_file_name)
        players_df = players_df.drop(columns=["Unnamed: 0"])

        # filter data to be only until the given season
        players_df = players_df[players_df["Season"].str.split("-").str[0].astype(int) < 2019]

        # TODO add changed club feature from notebook
        players_df["Season"] = players_df["Season"].str.replace("-", "-20")
        return players_df

    def get_games_df(self):
        return self.games_df

    def get_players_df(self):
        return self.players_df
