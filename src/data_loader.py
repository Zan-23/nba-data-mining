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

EVENTMSGACTIONTYPE_FIELD_GOAL_dict = {
    0: 'No Shot',
    1: 'Jump Shot',
    2: 'Running Jump Shot',
    3: 'Hook Shot',
    4: 'Tip Shot',
    5: 'Layup',
    6: 'Driving Layup',
    7: 'Dunk',
    8: 'Slam Dunk',
    9: 'Driving Dunk',
    40: 'Layup',
    41: 'Running Layup',
    42: 'Driving Layup',
    43: 'Alley Oop Layup',
    44: 'Reverse Layup',
    45: 'Jump Shot',
    46: 'Running Jump Shot',
    47: 'Turnaround Jump Shot',
    48: 'Dunk',
    49: 'Driving Dunk',
    50: 'Running Dunk',
    51: 'Reverse Dunk',
    52: 'Alley Oop Dunk',
    53: 'Tip Shot',
    54: 'Running Tip Shot',
    55: 'Hook Shot',
    56: 'Running Hook Shot',
    57: 'Driving Hook Shot',
    58: 'Turnaround Hook Shot',
    59: 'Finger Roll',
    60: 'Running Finger Roll',
    61: 'Driving Finger Roll',
    62: 'Turnaround Finger Roll',
    63: 'Fadeaway Jumper',
    64: 'Follow Up Dunk',
    65: 'Jump Hook Shot',
    66: 'Jump Bank Shot',
    67: 'Hook Bank Shot',
    71: 'Finger Roll Layup',
    72: 'Putback Layup',
    73: 'Driving Reverse Layup',
    74: 'Running Reverse Layup',
    75: 'Driving Finger Roll Layup',
    76: 'Running Finger Roll Layup',
    77: 'Driving Jump Shot',
    78: 'Floating Jump Shot',
    79: 'Pullup Jump Shot',
    80: 'Step Back Jump Shot',
    81: 'Pullup Bank Shot',
    82: 'Driving Bank Shot',
    83: 'Fadeaway Bank Shot',
    84: 'Running Bank Shot',
    85: 'Turnaround Bank Shot',
    86: 'Turnaround Fadeaway',
    87: 'Putback Dunk',
    88: 'Driving Slam Dunk',
    89: 'Reverse Slam Dunk',
    90: 'Running Slam Dunk',
    91: 'Putback Reverse Dunk',
    92: 'Putback Slam Dunk',
    93: 'Driving Bank Hook Shot',
    94: 'Jump Bank Hook Shot',
    95: 'Running Bank Hook Shot',
    96: 'Turnaround Bank Hook Shot',
    97: 'Tip Layup Shot',
    98: 'Cutting Layup Shot',
    99: 'Cutting Finger Roll Layup Shot',
    100: 'Running Alley Oop Layup Shot',
    101: 'Driving Floating Jump Shot',
    102: 'Driving Floating Bank Jump Shot',
    103: 'Running Pull-Up Jump Shot',
    104: 'Step Back Bank Jump Shot',
    105: 'Turnaround Fadeaway Bank Jump Shot',
    106: 'Running Alley Oop Dunk Shot',
    107: 'Tip Dunk Shot',
    108: 'Cutting Dunk Shot',
    109: 'Driving Reverse Dunk Shot',
    110: 'Running Reverse Dunk Shot'
}

EVENTMSGACTIONTYPE_FREE_THROW_dict = {
    10 : "Free Throw 1 of 1",
    11 : "Free Throw 1 of 2",
    12 : "Free Throw 2 of 2",
    13 : "Free Throw 1 of 3",
    14 : "Free Throw 2 of 3",
    15 : "Free Throw 3 of 3",
    16 : "Free Throw Technical",
    17 : "Free Throw Clear Path", #before 2006-07 season
    18 : "Free Throw Flagrant 1 of 2",
    19 : "Free Throw Flagrant 2 of 2",
    20 : "Free Throw Flagrant 1 of 1",
    21 : "Free Throw Technical 1 of 2",
    22 : "Free Throw Technical 2 of 2",
    23 : "Free Throw 1 of 2", #Guess
    24 : "Free Throw 2 of 2", #Guess
    25 : "Free Throw Clear Path 1 of 2", 
    26 : "Free Throw Clear Path 2 of 2",
    27 : "Free Throw Flagrant 1 of 3",
    28 : "Free Throw Flagrant 2 of 3",
    29 : "Free Throw Flagrant 3 of 3"
}

def load_data(columns = None, seasons = None, path = RAW_DATA_PATH, resolve = True):
    files = sorted([f for f in path.glob('*.csv')])
    season_dfs = []
    for f in files:
        #Only read specified seasons:
        if seasons and (f.stem[:7] not in seasons):
            continue
        season = pd.read_csv(f, usecols = columns)
        season["season_name"] = f.stem[:7]
        if resolve:
            if columns == None or "EVENTMSGTYPE" in columns:
                season["EVENTMSGTYPE"] = season["EVENTMSGTYPE"].replace(EVENTMSGTYPE_dict)
            if columns == None or "EVENTMSGACTIONTYPE" in columns:
                season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[1],"EVENTMSGACTIONTYPE"] = season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[1],"EVENTMSGACTIONTYPE"].replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
            if columns == None or "EVENTMSGACTIONTYPE" in columns:
                season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[2],"EVENTMSGACTIONTYPE"] = season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[2],"EVENTMSGACTIONTYPE"].replace(EVENTMSGACTIONTYPE_FIELD_GOAL_dict)
            if columns == None or "EVENTMSGACTIONTYPE" in columns:
                season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[3],"EVENTMSGACTIONTYPE"] = season.loc[season["EVENTMSGTYPE"]==EVENTMSGTYPE_dict[3],"EVENTMSGACTIONTYPE"].replace(EVENTMSGACTIONTYPE_FREE_THROW_dict)
        #remove empty columns:
        if "NEUTRALDESCRIPTION" in season.columns:
            season = season.drop("NEUTRALDESCRIPTION", 1)
        season_dfs.append(season)
    return pd.concat(season_dfs).reset_index(drop = True) if season_dfs else None