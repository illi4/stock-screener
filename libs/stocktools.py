# from libs.exceptions_lib import exception_handler
import json
import pandas as pd
import requests
import os
import time
from requests.exceptions import RequestException

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
            self.related_market_ticker = "ONEQ"
            self.stock_suffix = ''
        elif self.market_code == "ASX":
            self.exchange_url_part = "AU"
            self.related_market_ticker = "AXJO.INDX"
            self.stock_suffix = '.AU'
        elif self.market_code == "NYSE":
            self.exchange_url_part = "NYSE"
            self.related_market_ticker = "CETF"
            self.stock_suffix = ''

    def set_abbreviation(self, abbreviation):
        self.abbreviation = abbreviation


def get_industry_mapping(exchange):
    if exchange == "NASDAQ":
        return industry_mapping_nasdaq
    elif exchange == "ASX":
        return industry_mapping_asx


# Using proper api
def get_exchange_symbols(market_object, checked_workday, min_market_cap):
    global session

    stocks = []
    excluded_count = 0

    if session is None:
        session = requests.Session()

    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/{market_object.exchange_url_part}?api_token={eod_key}&fmt=json&filter=extended&date={checked_workday}"
    params = {"api_token": eod_key}
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        r = session.get(url, params=params)

        if r.status_code == 404:
            print(f"Ticker not found, skipping")
            return None, None

        if r.status_code == 502:
            print("Received status code 502, retrying...")
            time.sleep(1)
            attempt += 1
            continue

        if r.status_code != requests.codes.ok and r.status_code != 502:
            print(f"Status response is not Ok: {r.status_code}")
            exit(0)

        break
    else:
        print("Maximum attempts reached, exiting...")
        exit(0)

    data = json.loads(r.text)
    for elem in data:
        market_cap = elem.get("MarketCapitalization", 0)

        if market_cap >= min_market_cap:
            stock = dict(
                code=elem["code"],
                name=elem["name"],
                price=elem["close"],
                volume=elem["volume"],
                type=elem["type"],
                exchange=market_object.market_code,
                market_cap=market_cap
            )
            stocks.append(stock)
        else:
            excluded_count += 1

    print(f"Excluded {excluded_count} stocks due to market cap below {min_market_cap}")
    return stocks


# Add to stocktools.py
def get_earnings_calendar_non_selenium(date_from, date_to):
    """
    Fetch earnings calendar from StockTwits API for given date range

    Args:
        date_from (str): Start date in YYYY-MM-DD format
        date_to (str): End date in YYYY-MM-DD format

    Returns:
        set: Set of stock symbols that have earnings in the date range
    """
    global session

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
    }

    if session is None:
        session = requests.Session()

    url = f"https://api.stocktwits.com/api/2/discover/earnings_calendar?date_from={date_from}&date_to={date_to}"
    earnings_stocks = set()

    try:
        response = session.get(url, headers=headers)
        if response.status_code == 200:
            earnings_data = response.json()

            # Extract stock symbols for all dates in the earnings data
            if 'earnings' in earnings_data:
                for date, date_data in earnings_data['earnings'].items():
                    if 'stocks' in date_data:
                        for stock in date_data['stocks']:
                            earnings_stocks.add(stock['symbol'])

            print(f"Found {len(earnings_stocks)} stocks with earnings between {date_from} and {date_to}")
            return earnings_stocks

        else:
            print(f"Failed to fetch earnings data: {response.status_code}")
            return set()

    except Exception as e:
        print(f"Error fetching earnings data: {e}")
        return set()

def get_stock_data(code, reporting_date_start, max_retries=5, retry_delay=5):
    global session
    if session is None:
        session = requests.Session()

    #Note: cannot use the eod api endpoint because it is not split adjusted
    url = f"https://eodhd.com/api/technical/{code}?function=splitadjusted&api_token={eod_key}&order=a&fmt=json&from={reporting_date_start}"

    params = {"api_token": eod_key}

    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params)

            if r.status_code == 404:
                print(f"Ticker {code} not found, skipping")
                return None, None

            if r.status_code == requests.codes.ok:
                data = json.loads(r.text)
                df = pd.DataFrame.from_dict(data)

                if df.empty:
                    return None, None

                df = df[["date", "open", "high", "low", "close", "volume"]]
                df["date"] = pd.to_datetime(df["date"])
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

                return (
                    df[["timestamp", "open", "high", "low", "close"]],
                    df[["timestamp", "volume"]],
                )
            else:
                print(f"Attempt {attempt + 1}: Status response is not Ok: {r.status_code}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Exiting.")
                    return None, None

        except RequestException as e:
            print(f"Attempt {attempt + 1}: Request failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                return None, None

    return None, None

def get_earnings_calendar(date_from, date_to):
    '''
    Non selenium option stopped working
    '''
    chrome_options = Options()
    #chrome_options.add_argument('--headless')  # blocks me if I am using headless
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=chrome_options)

    # Execute script to hide webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    try:
        url = f"https://api.stocktwits.com/api/2/discover/earnings_calendar?date_from={date_from}&date_to={date_to}"
        print(f"Requesting: {url}")

        driver.get(url)

        # Wait for page to load and content to appear
        wait = WebDriverWait(driver, 10)

        # The API returns JSON directly, so we need to get the page source
        time.sleep(3)  # Give it time to load

        page_source = driver.page_source
        print("Page loaded successfully")

        # Try to find the JSON content
        try:
            # Method 1: Look for <pre> tag (common for JSON APIs)
            pre_element = driver.find_element(By.TAG_NAME, "pre")
            json_text = pre_element.text
        except:
            # Method 2: Get entire body text
            try:
                body_element = driver.find_element(By.TAG_NAME, "body")
                json_text = body_element.text
            except:
                # Method 3: Parse from page source
                # Look for JSON in page source
                start_idx = page_source.find('{"date_from"')
                if start_idx != -1:
                    # Find the end of JSON (look for closing brace)
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(page_source[start_idx:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = start_idx + i + 1
                                break
                    json_text = page_source[start_idx:end_idx]
                else:
                    print("Could not find JSON in page source")
                    return set()

        # Parse the JSON
        try:
            earnings_data = json.loads(json_text)
            print("JSON parsed successfully")

            # Extract stock symbols (same logic as your original function)
            earnings_stocks = set()
            if 'earnings' in earnings_data:
                for date, date_data in earnings_data['earnings'].items():
                    if 'stocks' in date_data:
                        for stock in date_data['stocks']:
                            earnings_stocks.add(stock['symbol'])

            print(f"Found {len(earnings_stocks)} stocks with earnings between {date_from} and {date_to}")
            return earnings_stocks

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw content: {json_text[:500]}...")  # Print first 500 chars for debugging
            return set()

    except Exception as e:
        print(f"Error with Selenium: {e}")
        return set()

    finally:
        driver.quit()

def ohlc_daily_to_weekly(df):
    df["start_of_week"] = df["timestamp"] - pd.to_timedelta(df["timestamp"].dt.dayofweek, unit='D')
    df_weekly = df.groupby(["start_of_week"]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "timestamp": "last"}
    )
    df_weekly = df_weekly.reset_index()
    df_weekly["year"] = df_weekly["timestamp"].dt.year
    df_weekly["timestamp"] = df_weekly["start_of_week"]
    df_weekly = df_weekly[["year", "timestamp", "open", "high", "low", "close", "start_of_week"]]
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
