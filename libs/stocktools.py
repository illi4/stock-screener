from selenium import webdriver
from bs4 import BeautifulSoup
import yfinance as yf
from libs.exceptions_lib import exception_handler
from libs.settings import asx_instruments_url

options = webdriver.ChromeOptions()

options.add_experimental_option('excludeSwitches', ['enable-logging'])  # removes the USB warning on Windows
options.add_argument(
    "--headless"
)  # headless means that no browser window is opened


def get_asx_symbols():
    driver = webdriver.Chrome(options=options)
    driver.get(asx_instruments_url)
    content = driver.page_source
    driver.close()

    soup = BeautifulSoup(content, 'html.parser')
    data = []
    table = soup.find('table', attrs={'class': 'mi-table mt-6'})
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [elem.text.strip() for elem in cols]
        data.append(cols)

    print(f"{len(data)} stocks retreived")

    stocks = [dict(code=elem[2], name=elem[3], price=float(elem[4].replace('$', ''))) for elem in data]
    return stocks

@exception_handler(handler_type="yfinance")
def get_stock_data(symbol):
    period = '300d'
    interval = '1d'

    asset = yf.Ticker(symbol)
    hist = asset.history(period=period, interval=interval).reset_index(drop=False)
    if hist.empty:
        print(f"Ticker {symbol} not found on Yahoo Finance")
        return None, None
    hist.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'div', 'splits']
    # For compatibility with the TA library
    return hist[['timestamp', 'open', 'high', 'low', 'close']], hist[['timestamp', 'volume']]

def ohlc_daily_to_weekly(df):
    df['week_number'] = df['timestamp'].dt.isocalendar().week
    df['year'] = df['timestamp'].dt.year
    df_weekly = df.groupby(['year', 'week_number']).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    # rename week to timestamp. even though it is not super correct, it's fine to do that for our purposes
    df_weekly = df_weekly.reset_index()
    df_weekly.columns = ['year', 'timestamp', 'open', 'high', 'low', 'close']
    return df_weekly


def ohlc_daily_to_monthly(df):
    df['month_number'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df_monthly = df.groupby(['year', 'month_number']).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    # rename week to timestamp. even though it is not super correct, it's fine to do that for our purposes
    df_monthly = df_monthly.reset_index()
    df_monthly.columns = ['year', 'timestamp', 'open', 'high', 'low', 'close']
    return df_monthly
