# from libs.exceptions_lib import exception_handler
import json
import pandas as pd
import requests
import os
import time

session = None  # to use in requests
eod_key = os.environ.get("API_KEY")

# Class for the market with its parameters
class Market:
    def __init__(self, market_code):
        self.market_code = market_code
        supported_markets = ['ASX', 'NASDAQ', 'NYSE']

        if self.market_code not in supported_markets:
            print(f"Supported markets are: {', '.join(supported_markets)}")

        if self.market_code == "NASDAQ":
            self.exchange_url_part = "NASDAQ"
            self.related_market_ticker = "ONEQ"  # nasdaq ETF
            self.stock_suffix = ''
        elif self.market_code == "ASX":
            self.exchange_url_part = "AU"
            self.related_market_ticker = "AXJO.INDX"
            self.stock_suffix = '.AU'
        elif self.market_code == "NYSE":
            self.exchange_url_part = "NYSE"
            self.related_market_ticker = "CETF"
            self.stock_suffix = ''

    def set_abbreviation(self, abbreviation):  # example of using stuff in classes
        self.abbreviation = abbreviation


def get_industry_mapping(exchange):
    if exchange == "NASDAQ":
        return industry_mapping_nasdaq
    elif exchange == "ASX":
        return industry_mapping_asx


# Using proper api
def get_exchange_symbols(market_object, checked_workday):
    global session

    stocks = []

    if session is None:
        session = requests.Session()

    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/{market_object.exchange_url_part}?api_token={eod_key}&fmt=json&filter=extended&date={checked_workday}"
    params = {"api_token": eod_key}
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        r = session.get(url, params=params)  # to speed things up

        if r.status_code == 404:
            print(f"Ticker not found, skipping")
            return None, None

        if r.status_code == 502:
            print("Received status code 502, retrying...")
            time.sleep(1)  # Sleep for 1 second
            attempt += 1
            continue

        if r.status_code != requests.codes.ok and r.status_code != 502:
            print(f"Status response is not Ok: {r.status_code}")
            exit(0)

        # Successful response, breaking out of the loop
        break
    else:
        print("Maximum attempts reached, exiting...")
        exit(0)

    data = json.loads(r.text)
    for elem in data:

        stock = dict(
            code=elem["code"],
            name=elem["name"],
            price=elem["close"],
            volume=elem["volume"],
            type=elem["type"],
            exchange=market_object.market_code,
        )
        stocks.append(stock)

    return stocks


def get_stock_data(code, reporting_date_start):
    global session
    if session is None:
        session = requests.Session()

    url = f"https://eodhistoricaldata.com/api/eod/{code}?api_token={eod_key}&order=a&fmt=json&from={reporting_date_start}"

    params = {"api_token": eod_key}
    r = session.get(url, params=params)  # to speed things up

    if r.status_code == 404:
        print(f"Ticker {code} not found, skipping")
        return None, None

    if r.status_code != requests.codes.ok:
        print(f"Status response is not Ok: {r.status_code}")
        exit(0)

    data = json.loads(r.text)

    df = pd.DataFrame.from_dict(
        data
    )  # question - will it give me data after 5pm on the curr day?

    # Could return an empty df - skip if so
    if df.empty:
        return None, None

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
    df["start_of_week"] = df["timestamp"] - pd.to_timedelta(df["timestamp"].dt.dayofweek, unit='D')
    df_weekly = df.groupby(["year", "week_number"]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "start_of_week": "first"}
    )
    # rename week to timestamp. even though it is not super correct, it's fine to do that for our purposes
    df_weekly = df_weekly.reset_index()
    df_weekly.columns = ["year", "timestamp", "open", "high", "low", "close", "start_of_week"]
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
