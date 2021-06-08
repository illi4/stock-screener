#from libs.exceptions_lib import exception_handler
from libs.settings import tzinfo, eod_key
import arrow
import json
import pandas as pd
import requests

session = None  # to use in requests


# Get the starting date for reporting
def get_data_start_date():
    current_date = arrow.now()
    shifted_date = current_date.shift(years=-1)
    data_start_date = shifted_date.format("YYYY-MM-DD")
    return data_start_date


data_start_date = get_data_start_date()

# Leaving as is, but this is not used anymore
industry_mapping_asx = {
    "Energy": "XEJ",
    "Basic Materials": "XMJ",
    "Industrials": "XNJ",
    "Consumer Cyclical": "XDJ",
    "Consumer Defensive": "XSJ",
    "Healthcare": "XHJ",
    "Financial Services": "XIJ",
    "Technology": "XFJ",
    "Communication Services": "XTJ",
    "Utilities": "XUJ",
    "Real Estate": "XPJ",
}

industry_mapping_nasdaq = {
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Technology": "XLK",
    "Communication Services": "XTL",
    "Utilities": "XLU",
    "Real Estate": "IYR",
}


def get_industry_mapping(exchange):
    if exchange == "NASDAQ":
        return industry_mapping_nasdaq
    elif exchange == "ASX":
        return industry_mapping_asx


def get_stock_suffix(exchange):
    # Defines stock suffix for YFinance
    if exchange == "NASDAQ":
        return ""
    elif exchange == "ASX":
        return ".AU"


def get_previous_workday():
    current_datetime = arrow.now()
    current_dow = current_datetime.isoweekday()
    if current_dow == 1:  # only subtract if today is Monday
        current_datetime = current_datetime.shift(days=-3)
    else:
        current_datetime = current_datetime.shift(days=-1)
    current_datetime = current_datetime.format("YYYY-MM-DD")
    return current_datetime


# Using proper api #HERE#
def get_exchange_symbols(exchange):
    stocks = []
    if exchange == "NASDAQ":
        exchange_url_part = "NASDAQ"
    elif exchange == "ASX":
        exchange_url_part = "AU"

    global session
    if session is None:
        session = requests.Session()

    previous_workday = get_previous_workday()

    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/{exchange_url_part}?api_token={eod_key}&fmt=json&filter=extended&date={previous_workday}"
    params = {"api_token": eod_key}
    r = session.get(url, params=params)  # to speed things up

    if r.status_code == 404:
        print(f"Error: not found")
        exit(0)

    if r.status_code != requests.codes.ok:
        print(f"Status response is not Ok: {r.status_code}")
        exit(0)

    data = json.loads(r.text)
    for elem in data:
        stock = dict(
            code=elem["code"],
            name=elem["name"],
            price=elem["close"],
            volume=elem["volume"],
            exchange=exchange,
        )
        stocks.append(stock)

    return stocks


def get_nasdaq_symbols():
    stocks = get_exchange_symbols("NASDAQ")
    return stocks


def get_asx_symbols():
    stocks = get_exchange_symbols("ASX")
    return stocks


def get_stock_data(code):
    global session
    if session is None:
        session = requests.Session()

    url = f"https://eodhistoricaldata.com/api/eod/{code}?api_token={eod_key}&order=a&fmt=json&from={data_start_date}"

    params = {"api_token": eod_key}
    r = session.get(url, params=params)  # to speed things up

    if r.status_code == 404:
        print(f"Ticker {code} not found, skipping")
        return None, None

    if r.status_code != requests.codes.ok:
        print(f"Status response is not Ok: {r.status_code}")
        exit(0)

    # This may happen if the stocks list is taken from a different location
    # if r.text == "[]":
    #    print("No recent stock data")
    #    exit(0)

    data = json.loads(r.text)

    df = pd.DataFrame.from_dict(
        data
    )  # question - will it give me data after 5pm on the curr day?
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # For compatibility with the TA library
    return (
        df[["timestamp", "open", "high", "low", "close"]],
        df[["timestamp", "volume"]],
    )


def ohlc_daily_to_weekly(df):
    df["week_number"] = df["timestamp"].dt.isocalendar().week
    df["year"] = df["timestamp"].dt.year
    df_weekly = df.groupby(["year", "week_number"]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    # rename week to timestamp. even though it is not super correct, it's fine to do that for our purposes
    df_weekly = df_weekly.reset_index()
    df_weekly.columns = ["year", "timestamp", "open", "high", "low", "close"]
    return df_weekly


def ohlc_daily_to_monthly(df):
    df["month_number"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df_monthly = df.groupby(["year", "month_number"]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    # rename week to timestamp. even though it is not super correct, it's fine to do that for our purposes
    df_monthly = df_monthly.reset_index()
    df_monthly.columns = ["year", "timestamp", "open", "high", "low", "close"]
    return df_monthly


def map_industry_code(industry_name):
    return industry_mapping_asx[industry_name]
