import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"

EVENTMSGTYPE_dict = {
    1 : "FIELD_GOAL_MADE",
    2 : "FIELD_GOAL_MISSED",
    3 : "FREE_THROW",
    4 : "REBOUND",
    5 : "TURNOVER",
    6 : "FOUL",
    7 : "VIOLATION",
    8 : "SUBSTITUTION",
    9 : "TIMEOUT",
    10 : "JUMP_BALL",
    11 : "EJECTION",
    12 : "PERIOD_BEGIN",
    13 : "PERIOD_END"
}

def load_data(columns = None, path = RAW_DATA_PATH):
    files = sorted([f for f in path.glob('*.csv')])
    seasons = []
    for f in files:
        season = pd.read_csv(f, usecols = columns)
        season["season_name"] = f.stem[:7]
        if columns == None or "EVENTMSGTYPE":
            season["EVENTMSGTYPE"] = season["EVENTMSGTYPE"].replace(EVENTMSGTYPE_dict)
        seasons.append(season)
    return pd.concat(seasons).reset_index(drop = True)