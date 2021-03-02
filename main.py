from libs.helpers import define_args
from libs.stocktools import get_asx_symbols
from libs.db import bulk_add_stocks, create_stock_table, delete_all_stocks
from libs.settings import price_min, price_max


def update_stocks():
    stocks = get_asx_symbols()
    create_stock_table()
    delete_all_stocks()
    print("Writing to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


if __name__ == "__main__":

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        print(f"Scanning the stocks priced {price_min} from to {price_max}")