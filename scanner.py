# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")
from collections import namedtuple

from libs.helpers import (
    define_args,
    dates_diff,
    format_number,
    get_previous_workday,
    get_current_workday,
    get_test_stocks,
    get_data_start_date,
)
from libs.signal import bullish_mri_based, market_bearish, bullish_anx_based
from libs.stocktools import (
    get_stock_data,
    ohlc_daily_to_weekly,
    get_exchange_symbols,
    Market
)
from libs.db import (
    bulk_add_stocks,
    create_stock_table,
    delete_all_stocks,
    get_stocks,
    get_update_date,
)
from libs.techanalysis import td_indicators, MA, fisher_distance
import pandas as pd
from time import time, sleep

from libs.read_settings import read_config
config = read_config()


def rewrite_stocks(exchange, stocks):
    create_stock_table()
    print(f"Deleting the existing stocks for {exchange}")
    delete_all_stocks(exchange)
    print(f"Writing info on {len(stocks)} stocks to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


def get_current_date():
    if arguments["date"] is None:
        if not config["locality"]["shift_update_day"]:
            current_date = get_current_workday()
        else:
            current_date = get_previous_workday()
            print(f'(i) Shifting update date to previous workday per config')
    else:
        current_date = arguments["date"].strftime("%Y-%m-%d")

    return current_date


def update_stocks(active_markets):

    checked_workday = get_current_date()
    print(f"Updating info on traded stocks as of {checked_workday}")

    for market in active_markets:
        print(f'Updating stock list for {market.market_code}')
        stocks = get_exchange_symbols(market, checked_workday)
        rewrite_stocks(market.market_code, stocks)

def check_update_date(active_markets):

    for market in active_markets:
        try:
            last_update_date = get_update_date(market.market_code)
            diff = dates_diff(last_update_date)
            if diff > 1:
                print(
                    "Warning: Stocks list was not updated today, the volume filter could work incorrectly. "
                    "Please consider running the --update first..."
                )
                sleep(3)
        except:
            print(
                "Warning: failed to check the update date for correctness"
            )

def last_volume_5D_MA(volume_daily):
    volume_ma_20 = MA(volume_daily, 20, colname="volume")
    return volume_ma_20["ma20"].iloc[-1]


def generate_indicators_daily_weekly(ohlc_daily):
    # Generates extra info from daily OHLC
    if len(ohlc_daily) < 8:
        print("Too recent asset, not enough daily data")
        return None, None
    else:
        td_values = td_indicators(ohlc_daily)
        ohlc_with_indicators_daily = pd.concat([ohlc_daily, td_values], axis=1)

    ohlc_weekly = ohlc_daily_to_weekly(ohlc_daily)
    if len(ohlc_weekly) < 8:
        print("Too recent asset, not enough weekly data")
        return None, None
    else:
        td_values_weekly = td_indicators(ohlc_weekly)
        ohlc_with_indicators_weekly = pd.concat([ohlc_weekly, td_values_weekly], axis=1)

    return ohlc_with_indicators_daily, ohlc_with_indicators_weekly


def report_on_shortlist(shortlist, exchange):
    checked_workday = get_current_date()
    print(
        f"{len(shortlist)} shortlisted stocks (sorted by 5-day MA vol) as of {checked_workday}:"
    )
    for stock in shortlist:
        print(f"{stock.code} ({stock.name}) | Volume {stock.volume} | fisherDaily {stock.fisherDaily:.2} | fisherWeekly {stock.fisherWeekly:.2}")


def process_data_at_date(ohlc_daily, volume_daily):
    # Removes most recent columns if there is an argument to look at a particular date
    # < in the condition because we assume that at a day we only have info on the previous day close
    if arguments["date"] is None:
        return ohlc_daily, volume_daily

    ohlc_daily_shifted = ohlc_daily[ohlc_daily["timestamp"] < arguments["date"]]
    volume_daily_shifted = volume_daily[volume_daily["timestamp"] < arguments["date"]]

    return ohlc_daily_shifted, volume_daily_shifted


def process_market_data_at_date(market_ohlc_daily, market_volume_daily):
    if arguments["date"] is None:
        return market_ohlc_daily, market_volume_daily

    market_ohlc_daily_shifted = market_ohlc_daily[market_ohlc_daily["timestamp"] < arguments["date"]]
    market_volume_daily_shifted = market_volume_daily[market_volume_daily["timestamp"] < arguments["date"]]

    return market_ohlc_daily_shifted, market_volume_daily_shifted


def calculate_extra_metrics(ohlc_with_indicators_daily, ohlc_with_indicators_weekly):
        metric_values = dict()

        metric_values['fisherDaily'] = fisher_distance(ohlc_with_indicators_daily).iloc[-1].values[0]
        metric_values['fisherWeekly'] = fisher_distance(ohlc_with_indicators_weekly).iloc[-1].values[0]

        return metric_values


def scan_stock(stocks, market, method): ###HERE###

    stock_suffix = market.stock_suffix
    shortlisted_stocks = []
    # placeholder for shortlisted stocks and their attributes
    # each stock will be a named tuple with the following definition:
    Stock = namedtuple('Stock', ['code', 'name', 'volume', 'fisherDaily', 'fisherWeekly'])

    # Iterate through the list of stocks
    for i, stock in enumerate(stocks):
        print(f"\n{stock.code} [{stock.name}] ({i + 1}/{len(stocks)})")
        ohlc_daily, volume_daily = get_stock_data(
            f"{stock.code}{stock_suffix}", reporting_date_start
        )
        if ohlc_daily is None:
            print("No data on the asset")
            continue  # skip this asset if there is no data

        ohlc_daily, volume_daily = process_data_at_date(ohlc_daily, volume_daily)

        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        if (
            ohlc_with_indicators_daily is None
            or ohlc_with_indicators_weekly is None
        ):
            continue


        #print(ohlc_with_indicators_daily.tail())
        #ohlc_with_indicators_weekly['fisher_dist'] = fisher_distance(ohlc_with_indicators_weekly)
        #print(ohlc_with_indicators_weekly.tail(10))


        # Check for confirmation depending on the method
        if method == 'mri':
            confirmation, _ = bullish_mri_based(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly,
                consider_volume_spike=True,
                output=True,
                stock_name=stock.name,
            )
        elif method == 'anx':
            confirmation, _ = bullish_anx_based(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly,
                output=True,
                stock_name=stock.name,
            )


        if confirmation:
            print(f"{stock.name} [v] meeting shortlisting conditions")
            volume_MA_5D = last_volume_5D_MA(volume_daily)

            if volume_MA_5D > config["filters"]["minimum_volume_level"]:
                print(
                    f'\n{stock.name} [v] meeting minimum volume level conditions '
                    f'({format_number(volume_MA_5D)} > {format_number(config["filters"]["minimum_volume_level"])})'
                )
                # Calculate extra metrics only for shortlisted stocks for a faster process
                metric_data = calculate_extra_metrics(ohlc_with_indicators_daily, ohlc_with_indicators_weekly)

                # Append the shortlist with a stock and its characteristics
                shortlisted_stocks.append(
                    Stock(code=stock.code,
                          name=stock.name,
                          volume=volume_MA_5D,
                          fisherDaily=metric_data['fisherDaily'],
                          fisherWeekly=metric_data['fisherWeekly'])
                )

            else:
                print(
                    f'\n{stock.name} [x] not meeting minimum volume level conditions '
                    f'({format_number(volume_MA_5D)} < {format_number(config["filters"]["minimum_volume_level"])})'
                )

        else:
            print(f"\n{stock.name} [x] not meeting shortlisting conditions")

    return shortlisted_stocks


def scan_exchange_stocks(market, method):
    # Check the market conditions
    market_ohlc_daily, market_volume_daily = get_stock_data(market.related_market_ticker, reporting_date_start)
    market_ohlc_daily, market_volume_daily = process_market_data_at_date(market_ohlc_daily, market_volume_daily)
    is_market_bearish, _ = market_bearish(market_ohlc_daily, market_volume_daily, output=True)

    if is_market_bearish:
        print("Overall market sentiment is bearish, not scanning individual stocks")
        exit(0)

    # Get the stocks for scanning
    if arguments["stock"] is None:
        stocks = get_stocks(
            exchange=market.market_code,
            price_min=config["pricing"]["min"],
            price_max=config["pricing"]["max"],
            min_volume=config["filters"]["minimum_volume_level"],
        )
    else:
        stocks = get_stocks(
            code=arguments["stock"]
        )

    # Limit per arguments as required
    if arguments["num"] is not None:
        print(f"Limiting to the first {arguments['num']} stocks")
        stocks = stocks[: arguments["num"]]

    total_number = len(stocks)
    print(
        f'Scanning {total_number} stocks priced {config["pricing"]["min"]} from to {config["pricing"]["max"]} '
        f'and with volume of at least {format_number(config["filters"]["minimum_volume_level"])}\n'
    )

    shortlist = scan_stock(stocks, market, method)  # this needs to be reworked to use named tuples

    # Sort the list by volume in decreasing order
    sorted_stocks = sorted(shortlist, key=lambda stock: stock.volume, reverse=True)

    return sorted_stocks


def scan_stocks(active_markets):

    # Create shortlists placeholder for each market
    shortlists = dict()

    for market in active_markets:
        shortlists[market.market_code] = scan_exchange_stocks(market, arguments["method"])  # need to pass the whole object

    for market in active_markets:
        print()
        print(f"Results for {market.market_code}")
        if len(shortlists[market.market_code]) > 0:
            report_on_shortlist(
                shortlists[market.market_code],
                market.market_code,
            )
        else:
            print(f"No shortlisted stocks for {market.market_code}")


if __name__ == "__main__":

    start_time = time()

    arguments = define_args()
    reporting_date_start = get_data_start_date(arguments["date"])

    # Initiate market objects
    active_markets = []
    for market_code in config["markets"]:
        active_markets.append(Market(market_code))

    if arguments["update"]:
        update_stocks(active_markets)
    if arguments["stock"]:
        print(f'Force checking one stock only: {arguments["stock"]}')

    if arguments["scan"]:
        check_update_date(active_markets)
        scan_stocks(active_markets)

    print()
    end_time = time()
    minutes_passed = (end_time - start_time) // 60
    print(f"{minutes_passed} minutes passed")
