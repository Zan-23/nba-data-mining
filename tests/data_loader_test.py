import os

import pandas as pd


def test_game_loader_equality():
    print("Starting test for game loader equality (from static filess)...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_org = pd.read_pickle(os.path.join(dir_path, "./../data/processed/load_data_games_arr_v1.pkl"))
    data_zan = pd.read_pickle(os.path.join(dir_path, "./../data/processed/load_data_games_arr_v2_zan.pkl"))

    # compare the two methods for any different data
    extra_columns = ["games_already_played_in_season", "home_players_deployed_ids",
                     "visitor_players_deployed_ids",
                     "home_TSP", "visitor_TSP",
                     "home_final_score_diff", "visitor_final_score_diff"]
    common_cols = [i for i in data_zan.columns if i not in extra_columns]

    data_zan = data_zan[common_cols]
    difference_mask = data_org[data_org.index.isin(data_zan.index)].sort_index(axis=1) == data_zan.sort_index(axis=1)
    wrong_data_rows = data_zan.sort_index(axis=1)[~difference_mask].dropna(axis=1)

    assert len(wrong_data_rows.columns) <= 1, f"There are {len(wrong_data_rows.columns)} columns with wrong data!"


if __name__ == "__main__":
    test_game_loader_equality()