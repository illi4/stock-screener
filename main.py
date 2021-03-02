from libs.helpers import define_args, dates_diff
from libs.stocktools import get_asx_symbols
from libs.db import bulk_add_stocks, create_stock_table, delete_all_stocks, get_stocks, get_update_date
from libs.settings import price_min, price_max


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
    stocks = get_stocks(price_min=price_min, price_max=price_max)
    total_number = len(stocks)
    print(total_number)


if __name__ == "__main__":

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        print(f"Scanning the stocks priced {price_min} from to {price_max}")
        check_update_date()
        scan_stocks()