from selenium import webdriver
from bs4 import BeautifulSoup
import yfinance as yf
from libs.exceptions_lib import exception_handler
from libs.settings import (
    asx_instruments_url,
    tzinfo,
    asx_stock_url,
    nasdaq_instruments_url,
)
import arrow
import string
import concurrent.futures
import numpy as np
import itertools

options = webdriver.ChromeOptions()

options.add_experimental_option(
    "excludeSwitches", ["enable-logging"]
)  # removes the USB warning on Windows
options.add_argument("--headless")  # headless means that no browser window is opened

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


def get_stocks_per_letter(letters, exchange, exchange_url):
    driver = webdriver.Chrome(options=options)
    stocks = []

    for letter in letters:
        url = f"{exchange_url}/{letter.upper()}.htm"
        print(f"Processing {url}")
        driver.get(url)
        content = driver.page_source

        soup = BeautifulSoup(content, "html.parser")
        data = []
        table = soup.find("table", attrs={"class": "quotes"})
        table_body = table.find("tbody")

        rows = table_body.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            cols = [elem.text.strip() for elem in cols]
            data.append(cols)

        for elem in data:
            # first one is an empty string
            if len(elem) > 0:
                stocks.append(
                    dict(
                        code=elem[0],
                        name=elem[1],
                        price=float(elem[4].replace(",", "")),
                        volume=float(elem[5].replace(",", "")),
                        exchange=exchange,
                    )
                )

    driver.close()
    return stocks


def get_exchange_symbols(exchange):
    all_letters = list(string.ascii_lowercase)

    if exchange == "NASDAQ":
        exchange_url = nasdaq_instruments_url
    elif exchange == "ASX":
        exchange_url = asx_instruments_url
        # ASX also has some numerical values
        for extra_symbol in ["1", "2", "3", "4", "5", "8", "9"]:
            all_letters.append(extra_symbol)

    # Threading
    letters_groups = np.array_split(all_letters, 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(get_stocks_per_letter, letter_set, exchange, exchange_url)
            for set_counter, letter_set in enumerate(letters_groups)
        ]
        stocks_info = [f.result() for f in futures]

    # Join list of lists into a single list
    stocks = list(itertools.chain.from_iterable(stocks_info))

    return stocks


def get_nasdaq_symbols():
    stocks = get_exchange_symbols("NASDAQ")
    return stocks


def get_asx_symbols():
    stocks = get_exchange_symbols("ASX")
    return stocks


def ohlc_last_day_workaround(df):
    # Need to check whether the last day is today and remove if so
    # Due to YFinance bug - not showing the right data for the current day
    hist_last_day = df["Date"].iloc[-1]
    hist_last_day = arrow.get(hist_last_day)
    hist_last_day = hist_last_day.replace(tzinfo=tzinfo)
    current_date = arrow.now()
    if hist_last_day.format("YYYY-MM-DD") == current_date.format("YYYY-MM-DD"):
        df.drop(df.tail(1).index, inplace=True)
    return df


@exception_handler(handler_type="yfinance")
def get_stock_data(symbol):
    period = "300d"
    interval = "1d"

    asset = yf.Ticker(symbol)
    hist = asset.history(period=period, interval=interval).reset_index(drop=False)

    if hist.empty:
        print(f"Ticker {symbol} not found on Yahoo Finance")
        return None, None

    hist = ohlc_last_day_workaround(hist)
    hist.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "div",
        "splits",
    ]

    # For compatibility with the TA library
    return (
        hist[["timestamp", "open", "high", "low", "close"]],
        hist[["timestamp", "volume"]],
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


@exception_handler(handler_type="yfinance")
def get_industry(code, exchange):
    # Suffix definition
    if exchange == "ASX":
        stock_suffix = ".AX"
    else:
        stock_suffix = ""

    # Getting sector for a stock using YFinance
    asset = yf.Ticker(f"{code}{stock_suffix}")

    try:
        info = asset.info
        # May not always be present in the data
        if "sector" in info.keys():
            industry = info["sector"]
        else:
            industry = "-"
    except ValueError:
        print(f"Cannot get data for {code}{stock_suffix}")
        industry = "-"

    return industry


def get_industry_from_web(code, driver):
    # To be used in batch by get_industry_from_web_batch
    # Works for asx only
    driver.get(f"{asx_stock_url}/{code}")
    content = driver.page_source

    soup = BeautifulSoup(content, "html.parser")
    data = []

    h_key = soup.find("h3", string="Key Information")
    if h_key is None:
        return None

    table = h_key.find_next("table", attrs={"class": "mi-table"})
    table_body = table.find("tbody")

    rows = table_body.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [elem.text.strip() for elem in cols]
        data.append(cols)

    data = {item[0]: item[1] for item in data}
    return data["Sector"]


def get_industry_from_web_batch(codes):
    # To be used instead of YFinance as this is almost twice faster
    # Works for asx only
    response = dict()
    driver = webdriver.Chrome(options=options)
    for code in codes:
        response[code] = get_industry_from_web(code, driver)
    driver.close()
    return response


def map_industry_code(industry_name):
    return industry_mapping_asx[industry_name]
