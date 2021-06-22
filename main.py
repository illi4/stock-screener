from libs.helpers import (
    define_args,
    dates_diff,
    format_number,
    get_previous_workday,
    get_test_stocks,
    get_data_start_date,
)
from libs.signal import bullish_ma_based
from libs.stocktools import (
    get_asx_symbols,
    get_nasdaq_symbols,
    get_stock_data,
    ohlc_daily_to_weekly,
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


def rewrite_stocks(exchange, stocks):
    create_stock_table()
    print(f"Deleting the existing stocks for {exchange}")
    delete_all_stocks(exchange)
    print(f"Writing info on {len(stocks)} stocks to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


def update_stocks():
    exchange = arguments["exchange"]

    if arguments["date"] is None:
        checked_workday = get_previous_workday()
    else:
        checked_workday = arguments["date"].strftime("%Y-%m-%d")

    print(f"Updating info on traded stocks as of {checked_workday}")

    if exchange == "ASX":
        stocks = get_asx_symbols(checked_workday)
        rewrite_stocks(exchange, stocks)
    elif exchange == "NASDAQ":
        stocks = get_nasdaq_symbols(checked_workday)
        rewrite_stocks(exchange, stocks)
    elif exchange == "ALL":
        for each_exchange in ["ASX", "NASDAQ"]:
            print(f"Processing {each_exchange}...")
            stocks = get_exchange_symbols(each_exchange, checked_workday)
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


def report_on_shortlist(shortlist, industry_score, exchange):
    if arguments["date"] is None:
        as_of = "today"
    else:
        as_of = arguments["date"].strftime("%Y-%m-%d")

    print(
        f"{len(shortlist)} shortlisted stocks (sorted by 5-day MA vol) as of {as_of}:"
    )
    for stock in shortlist:
        print(f"{stock[0]} ({stock[1]}) | {format_number(stock[2])} volume")


def process_data_at_date(ohlc_daily, volume_daily):
    # Removes most recent columns if there is an argument to look at a particular date
    # < in the condition because we assume that at a day we only have info on the previous day close
    if arguments["date"] is None:
        return ohlc_daily, volume_daily

    ohlc_daily_shifted = ohlc_daily[ohlc_daily["timestamp"] < arguments["date"]]
    volume_daily_shifted = volume_daily[volume_daily["timestamp"] < arguments["date"]]

    return ohlc_daily_shifted, volume_daily_shifted


def scan_stock(stocks, exchange):
    stock_suffix = get_stock_suffix(exchange)

    try:
        shortlisted_stocks = []
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

            confirmation, _ = bullish_ma_based(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly,
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

    except KeyboardInterrupt:
        print("KeyboardInterrupt: exiting")
        exit(0)


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

    # Get industry bullishness scores: disabled as it was not helpful
    # industry_momentum, industry_score = get_industry_momentum(exchange)
    industry_momentum, industry_score = None, None

    total_number = len(stocks)
    print(
        f"Scanning {total_number} stocks priced {price_min} from to {price_max} "
        f"and with volume of at least {format_number(minimum_volume_level)}\n"
    )

    # Back to basics because parallel run shows incorrect results for some reason
    shortlist = scan_stock(stocks, exchange)

    # Short the stocks by volume desc
    sorted_stocks = sorted(shortlist, key=lambda tup: tup[2], reverse=True)
    shortlist = [(stock[0], stock[1], stock[2]) for stock in sorted_stocks]

    return shortlist, industry_momentum, industry_score


def scan_stocks():

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
    reporting_date_start = get_data_start_date(arguments["date"])

    if arguments["update"]:
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()

    print()
    end_time = time()
    minutes_passed = (end_time - start_time) // 60
    print(f"{minutes_passed} minutes passed")
