from libs.helpers import define_args, dates_diff, format_number, get_test_stocks
from libs.signal import bullish_ma_based
from libs.stocktools import (
    get_asx_symbols,
    get_nasdaq_symbols,
    get_stock_data,
    ohlc_daily_to_weekly,
    get_industry_mapping,
    get_industry,
    get_exchange_symbols,
    get_stock_suffix,
)
from libs.db import (
    bulk_add_stocks,
    create_stock_table,
    delete_all_stocks,
    get_stocks,
    get_update_date,
)
from libs.settings import price_min, price_max, minimum_volume_level
from libs.techanalysis import td_indicators, MA
import pandas as pd
from time import time, sleep
import concurrent.futures
import numpy as np
import itertools


def rewrite_stocks(exchange, stocks):
    create_stock_table()
    print(f"Deleting the existing stocks for {exchange}")
    delete_all_stocks(exchange)
    print(f"Writing info on {len(stocks)} stocks to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


def update_stocks():
    exchange = arguments["exchange"]
    if exchange == "ASX":
        stocks = get_asx_symbols()
        rewrite_stocks(exchange, stocks)
    elif exchange == "NASDAQ":
        stocks = get_nasdaq_symbols()
        rewrite_stocks(exchange, stocks)
    elif exchange == "ALL":
        for each_exchange in ["ASX", "NASDAQ"]:
            print(f"Updating {each_exchange}")
            stocks = get_exchange_symbols(each_exchange)
            rewrite_stocks(each_exchange, stocks)


def check_update_date():
    last_update_date = get_update_date()
    diff = dates_diff(last_update_date)
    if diff > 1:
        print(
            "Warning: Stocks list was not updated today, the volume filter could work incorrectly. "
            "Please consider running the --update first..."
        )
        sleep(3)


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


def get_industry_momentum(exchange):
    print(f"Calculating industry momentum scores for {exchange}...")
    industry_momentum, industry_score = dict(), dict()
    industry_mapping = get_industry_mapping(exchange)

    if exchange == "ASX":
        stock_prefix = "^A"
    else:
        stock_prefix = ""

    ma_num = ma_num_per_system()

    for name, code in industry_mapping.items():
        ohlc_daily, volume_daily = get_stock_data(f"{stock_prefix}{code}")
        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        industry_momentum[code], industry_score[code] = bullish_ma_based(
            ohlc_with_indicators_daily,
            volume_daily,
            ohlc_with_indicators_weekly,
            ma_num,
            consider_volume_spike=False,
            output=False,
        )
    return industry_momentum, industry_score


def report_on_shortlist(
    shortlist, industry_score, exchange, report_on_industry=True, print_progress=False
):
    if report_on_industry:
        # Get the sectors for shortlisted stocks only
        print(
            f"Getting industry data for {len(shortlist)} shortlisted stocks, hold on..."
        )
        # Get stock codes to collect industries
        stock_codes = [stock[0] for stock in shortlist]
        sectors = dict()
        for idx, stock_code in enumerate(stock_codes):
            if print_progress:
                print(f"- {stock_code} ({idx + 1}/{len(stock_codes)})")
            sectors[stock_code] = get_industry(stock_code, exchange=exchange)

        industry_mapping = get_industry_mapping(exchange)

        print()
        print(f"All shortlisted stocks (sorted by 5-day moving average volume):")
        for stock in shortlist:

            # May not find a sector for a stock
            if sectors[stock[0]] == "-":
                print(
                    f"- {stock[0]} ({stock[1]}) | {format_number(stock[2])} vol | "
                    f"Sector score unavailable"
                )
            else:
                industry_code = industry_mapping[sectors[stock[0]]]
                print(
                    f"- {stock[0]} ({stock[1]}) | {format_number(stock[2])} vol | "
                    f"{sectors[stock[0]]} score {industry_score[industry_code]}/5"
                )
    else:
        print(f"All shortlisted stocks (sorted by 5-day moving average volume):")
        for stock in shortlist:
            print(f"- {stock[0]} ({stock[1]}) | {format_number(stock[2])} vol")


def ma_num_per_system():
    system_name = arguments["system"]
    if system_name == "2ma":
        ma_num = 2
    elif system_name == "3ma":
        ma_num = 3
    return ma_num


def scan_stock_group(stocks, set_counter, exchange):
    stock_suffix = get_stock_suffix(exchange)
    ma_num = ma_num_per_system()

    shortlisted_stocks = []
    for i, stock in enumerate(stocks):
        print(
            f"\n{stock.code} [{stock.name}] ({i + 1}/{len(stocks)}) [thread {set_counter + 1}]"
        )
        ohlc_daily, volume_daily = get_stock_data(f"{stock.code}{stock_suffix}")

        if ohlc_daily is None:
            print("No data on the asset")
            continue  # skip this asset if there is no data

        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        if ohlc_with_indicators_daily is None or ohlc_with_indicators_weekly is None:
            continue

        confirmation, _ = bullish_ma_based(
            ohlc_with_indicators_daily,
            volume_daily,
            ohlc_with_indicators_weekly,
            ma_num,
            consider_volume_spike=True,
            output=True,
            stock_name=stock.name,
        )
        if confirmation:
            print(f"{stock.name} [v] meeting shortlisting conditions")
            volume_MA_5D = last_volume_5D_MA(volume_daily)

            if volume_MA_5D > minimum_volume_level:
                print(
                    f"\n{stock.name} [v] meeting minimum volume level conditions "
                    f"({format_number(volume_MA_5D)} > {format_number(minimum_volume_level)})"
                )
                shortlisted_stocks.append((stock.code, stock.name, volume_MA_5D))
            else:
                print(
                    f"\n{stock.name} [x] not meeting minimum volume level conditions "
                    f"({format_number(volume_MA_5D)} < {format_number(minimum_volume_level)})"
                )

        else:
            print(f"\n{stock.name} [x] not meeting shortlisting conditions")

    return shortlisted_stocks


def scan_exchange_stocks(exchange):
    stocks = get_stocks(
        exchange=exchange,
        price_min=price_min,
        price_max=price_max,
        min_volume=minimum_volume_level,
    )

    # Limit per arguments as required
    if arguments["num"] is not None:
        print(f"Limiting to the first {arguments['num']} stocks")
        stocks = stocks[: arguments["num"]]

    # Get industry bullishness scores
    industry_momentum, industry_score = get_industry_momentum(exchange)

    total_number = len(stocks)
    print(
        f"Scanning {total_number} stocks priced {price_min} from to {price_max} "
        f"and with volume of at least {format_number(minimum_volume_level)}\n"
    )

    # Split stocks on 5 parts for threading
    # It is MUCH faster with threading
    # KeyboardInterrupt does not work very well. Did not find a solution just yet which is ok.
    stocks_sets = np.array_split(stocks, 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(scan_stock_group, stocks_set, set_counter, exchange)
            for set_counter, stocks_set in enumerate(stocks_sets)
        ]
        shortlisted_stock_collections = [f.result() for f in futures]

    # Join list of lists into a single list
    shortlist = list(itertools.chain.from_iterable(shortlisted_stock_collections))

    # Short the stocks by volume desc
    sorted_stocks = sorted(shortlist, key=lambda tup: tup[2], reverse=True)
    shortlist = [(stock[0], stock[1], stock[2]) for stock in sorted_stocks]

    return shortlist, industry_momentum, industry_score


def scan_stocks():
    if arguments["system"] is None:
        print("The system name must be specified for a scan")
        exit(0)

    if arguments["exchange"] != "ALL":
        shortlist, industry_momentum, industry_score = scan_exchange_stocks(
            arguments["exchange"]
        )
        print()
        if len(shortlist) > 0:
            report_on_shortlist(shortlist, industry_score, arguments["exchange"])
        else:
            print(f"No shortlisted stocks")
    else:
        all_exchanges = ["ASX", "NASDAQ"]
        shortlists, industry_momentums, industry_scores = dict(), dict(), dict()
        for each_exchange in all_exchanges:
            (
                shortlists[each_exchange],
                industry_momentums[each_exchange],
                industry_scores[each_exchange],
            ) = scan_exchange_stocks(each_exchange)

        for each_exchange in all_exchanges:
            print()
            print(f"Results for {each_exchange}")
            if len(shortlists[each_exchange]) > 0:
                report_on_shortlist(
                    shortlists[each_exchange],
                    industry_scores[each_exchange],
                    each_exchange,
                )
            else:
                print(f"No shortlisted stocks")


if __name__ == "__main__":

    start_time = time()

    arguments = define_args()
    if arguments["update"]:
        print(
            f"Updating the stocks list (exchange: {arguments['exchange'].capitalize()})"
        )
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()

    print()
    end_time = time()
    minutes_passed = (end_time - start_time) // 60
    print(f"{minutes_passed} minutes passed")
