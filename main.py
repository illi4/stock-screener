from libs.helpers import define_args, dates_diff
from libs.stocktools import get_asx_symbols, get_stock_data, ohlc_daily_to_weekly
from libs.db import bulk_add_stocks, create_stock_table, delete_all_stocks, get_stocks, get_update_date
from libs.settings import price_min, price_max
from libs.techanalysis import td_indicators, MA
import pandas as pd


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
        print("Warning: Stocks list updated more than 5 days ago")


def scan_stocks():
    shortlisted_stocks = []
    stocks = get_stocks(price_min=price_min, price_max=price_max)
    total_number = len(stocks)
    print(f"Scanning {total_number} stocks priced {price_min} from to {price_max}")

    for i, stock in enumerate(stocks):
        print(f"{stock.code} [{stock.name}] ({i+1}/{len(stocks)})")
        ohlc_daily, volume_daily = get_stock_data(f"{stock.code}.AX")

        if ohlc_daily is None:
            print("No data on the asset")
            continue  # skip if no data

        if len(ohlc_daily) < 8:
            print("Too recent asset, not enough daily data")
            continue

        td_values = td_indicators(ohlc_daily)
        ohlc_with_indicators_daily = pd.concat([ohlc_daily, td_values], axis=1)

        ohlc_weekly = ohlc_daily_to_weekly(ohlc_daily)
        if len(ohlc_weekly) < 8:
            print("Too recent asset, not enough weekly data")
            continue

        td_values_weekly = td_indicators(ohlc_weekly)
        ohlc_with_indicators_weekly = pd.concat([ohlc_weekly, td_values_weekly], axis=1)

        #print(ohlc_with_indicators_weekly)

        # CONTINUE
        # Also need to get avg 5-d volume to sort by volume and to detect if there were spikes in volume over the last X days

        exit(0)


if __name__ == "__main__":

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()