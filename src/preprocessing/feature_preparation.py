import os
import pickle

import pandas as pd

from src.data_loader import load_game_data_zan


def prepare_features_for_season_based_predictions(save_to_file=False):
    """
    Function will prepare the dataset and all its features for the season based predictions.
    :return:
    """
    player_info = prepare_player_df()
    games_df = load_game_data_zan(force_recompute=False)
    games_df["season_name"] = games_df["season_name"].str.replace("-", "-20")
    player_info["Season"] = player_info["Season"].str.replace("-", "-20")

    general_nba_info_dict = {
        "all_seasons_info": {
            "team_ids_arr": list(set(games_df["visitor_team_id"]).union(games_df["home_team_id"])),
            "player_ids_arr": list(player_info["Player ID"].unique()),
            "games_played": len(games_df.index),
        },
        "seasons":
            dict.fromkeys(list(games_df["season_name"].unique()), {"games": None, "player_info": None}),
    }

    for s_name in games_df["season_name"].unique():
        general_nba_info_dict["seasons"][str(s_name)]["games"] = games_df[games_df["season_name"] == s_name]
        general_nba_info_dict["seasons"][str(s_name)]["player_info"] = player_info[player_info["Season"] == s_name]

    if save_to_file:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "./../../data/processed/prepared_predictions_dict.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(general_nba_info_dict, f)

    return general_nba_info_dict


def prepare_player_df():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "./../../data/raw/player-data/player_info.csv")
    print(file_path)
    player_df = pd.read_csv(file_path)
    player_df = player_df.drop(columns=["Unnamed: 0"])

    # filter data to be only until the given season
    player_df = player_df[player_df["Season"].str.split("-").str[0].astype(int) < 2019]

    # TODO add changed club feature from notebook

    return player_df

