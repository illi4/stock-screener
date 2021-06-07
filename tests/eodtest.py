import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # to access the parent dir

from libs.stocktools import eod_data_key

import json
from urllib.request import urlopen
import pandas as pd

def get_eod_data(code):
    url = f"https://eodhistoricaldata.com/api/eod/{code}?api_token={eod_data_key}&order=a&fmt=json"

    with urlopen(url) as response:
        data = json.loads(response.read())

    df = pd.DataFrame.from_dict(data)  # question - will it give me data after 5pm on the curr day? YES it does
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # For compatibility with the TA library
    return (
        df[["timestamp", "open", "high", "low", "close"]],
        df[["timestamp", "volume"]],
    )

#a, b = get_eod_data("IMU.AU")
#a, b = get_eod_data("TSLA")
a, b = get_eod_data("POD.AU")
print(a)