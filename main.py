from libs.helpers import define_args, dates_diff, format_number, get_test_stocks
from libs.criteria import met_conditions_bullish
from libs.stocktools import (
    get_asx_symbols,
    get_stock_data,
    ohlc_daily_to_weekly,
    industry_mapping,
    get_industry_from_web_batch,
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
from time import time


def update_stocks():
    stocks = get_asx_symbols()
    create_stock_table()
    delete_all_stocks()
    print("Writing to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


def check_update_date():
    last_update_date = get_update_date()
    diff = dates_diff(last_update_date)
    if diff > 5:
        print("Stocks list updated more than 5 days ago, please run the --update first")
        exit(0)


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


def get_industry_momentum():
    print("Calculating industry momentum scores...")
    industry_momentum, industry_score = dict(), dict()
    for name, code in industry_mapping.items():
        ohlc_daily, volume_daily = get_stock_data(f"^A{code}")
        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        industry_momentum[code], industry_score[code] = met_conditions_bullish(
            ohlc_with_indicators_daily,
            volume_daily,
            ohlc_with_indicators_weekly,
            consider_volume_spike=False,
            output=False,
        )
    return industry_momentum, industry_score


def report_on_shortlist(shortlist, industry_score):
    # Get the sectors for shortlisted stocks only
    print(f"Getting industry data for {len(shortlist)} shortlisted stocks, hold on...")

    # Get stock codes to collect industries
    stock_codes = [stock[0] for stock in shortlist]
    sectors = get_industry_from_web_batch(stock_codes)

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


def scan_stocks():
    shortlisted_stocks = []
    stocks = get_stocks(price_min=price_min, price_max=price_max)

    # Limit per arguments as required
    if arguments["num"] is not None:
        print(f"Limiting to the first {arguments['num']} stocks")
        stocks = stocks[: arguments["num"]]

    # Get industry bullishness scores
    industry_momentum, industry_score = get_industry_momentum()

    total_number = len(stocks)
    print(f"Scanning {total_number} stocks priced {price_min} from to {price_max}")

    for i, stock in enumerate(stocks):
        print(f"{stock.code} [{stock.name}] ({i + 1}/{len(stocks)})")
        ohlc_daily, volume_daily = get_stock_data(f"{stock.code}.AX")

        if ohlc_daily is None:
            print("No data on the asset")
            continue  # skip this asset if there is no data

        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        if ohlc_with_indicators_daily is None or ohlc_with_indicators_weekly is None:
            continue

        confirmation, _ = met_conditions_bullish(
            ohlc_with_indicators_daily,
            volume_daily,
            ohlc_with_indicators_weekly,
            consider_volume_spike=True,
            output=True,
        )
        if confirmation:
            print("- [v] meeting shortlisting conditions")
            volume_MA_5D = last_volume_5D_MA(volume_daily)

            if volume_MA_5D > minimum_volume_level:
                print(
                    f"- [v] meeting minimum volume level conditions "
                    f"({format_number(volume_MA_5D)} > {format_number(minimum_volume_level)})"
                )
                shortlisted_stocks.append((stock.code, stock.name, volume_MA_5D))
            else:
                print(
                    f"- [x] not meeting minimum volume level conditions "
                    f"({format_number(volume_MA_5D)} < {format_number(minimum_volume_level)})"
                )

        else:
            print("- [x] not meeting shortlisting conditions")

    # Sort by volume (index 2) descending
    sorted_stocks = sorted(shortlisted_stocks, key=lambda tup: tup[2], reverse=True)
    shortlist = [(stock[0], stock[1], stock[2]) for stock in sorted_stocks]

    print()
    if len(shortlist) > 0:
        report_on_shortlist(shortlist, industry_score)

    else:
        print(f"No shortlisted stocks")


if __name__ == "__main__":

    start_time = time()

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()

    print()
    end_time = time()
    minutes_passed = (end_time - start_time) // 60
    print(f"{minutes_passed} minutes passed")
