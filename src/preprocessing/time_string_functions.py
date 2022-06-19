from datetime import timedelta

import numpy as np
import pandas as pd


def wrong_timestamps_removal(df):
    """
    TODO add documentation
    Used for finding the wrong timestamps, mostly before 2012 since there shouldn't be any left after 2012
    :return:
    """
    # removing wrongly formatted timestamps, relavant for data before 2012
    correct_time_format_regex = "^\s?([0][0-9]|[1][0-2]|[0-9]):[0-5][0-9]\s?(?:AM|PM|am|pm)$"
    correct_t_mask = df["WCTIMESTRING"].str.match(correct_time_format_regex)
    correct_t_df = df[correct_t_mask]
    wrong_t_df = df[~correct_t_mask]

    print(f"{len(wrong_t_df)}/{len(df)} rows will be discarded due to wrong timestamps.")


def convert_wctimestring_to_datetime(df):
    print("Starting timestamp conversion ...")

    # remove rows from seasons before 2012
    after_2012_data = df[df["season_name"].str.split("-").str[0].astype(int) >= 2012]

    # convert the dataframe to correct datetime format
    after_2012_data["real_world_time"] = pd.to_datetime(after_2012_data["WCTIMESTRING"], format='%I:%M %p')\
        .dt.strftime('%H:%M')
    # group data by game id, and add new columns
    grouped_game_id_df = after_2012_data.groupby("GAME_ID").agg({"real_world_time": list})
    grouped_game_id_df["game_start_time"] = np.nan
    grouped_game_id_df["game_end_time"] = np.nan

    print("Finished grouping data, starting extraction of start and end game times ...")
    for index, row in grouped_game_id_df.iterrows():
        until_midnight = []
        after_midnight = []

        for time_str in row["real_world_time"]:
            # very dumb handling of edge cases and wrong values
            if "23:59" >= time_str >= "10:59":
                until_midnight.append(time_str)
            else:
                after_midnight.append(time_str)
        curr_row_arr = sorted(until_midnight) + sorted(after_midnight)

        start_time = pd.to_datetime(curr_row_arr[0], format='%H:%M')
        end_time = pd.to_datetime(curr_row_arr[-1], format='%H:%M')

        if start_time.hour > 10 and end_time.hour < 10:
            # add one day, if clock goes over midnight
            end_time = end_time + timedelta(hours=24)
        grouped_game_id_df.loc[index, "game_start_time"] = start_time
        grouped_game_id_df.loc[index, "game_end_time"] = end_time

    # get game duration and convert is to minutes
    grouped_game_id_df["game_duration"] = (grouped_game_id_df["game_end_time"] - grouped_game_id_df["game_start_time"])\
        .dt.total_seconds().div(60).astype(int)

    # converting the start times, to just dates without date
    grouped_game_id_df["game_start_time"] = [val.time() for val in grouped_game_id_df["game_start_time"]]
    grouped_game_id_df["game_end_time"] = [val.time() for val in grouped_game_id_df["game_end_time"]]

    # TODO replace wrong values
    # games_with_broken_t_arr = []
    # handle_wrong_timestamps_after_2012()
    return grouped_game_id_df


def handle_wrong_timestamps_after_2012(df):
    game_id_arr_corrupt = [21400968, 21600179, 21700025, 21700077, 21700097, 21700187, 21700346, 21700871, 21701054]
    # TODO add automatic fixing

