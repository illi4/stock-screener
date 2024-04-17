# Fills in the missing prices for paper entries where applicable using daily data
# Intraday is not implemented because of the higher cost for intraday data

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

import libs.gsheetobj as gsheetsobj
from libs.helpers import get_data_start_date, define_args_method_only
from libs.stocktools import get_stock_data, Market
import arrow

from libs.read_settings import read_config
config = read_config()

reporting_date_start = get_data_start_date()

def fill_prices():

    sheet_name = config["logging"]["gsheet_name"]
    tab_name = config["logging"]["gsheet_tab_name"]

    ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)

    for index, row in ws.iterrows():
        if (
            row["Trade type"] == "paper"
            and row["Entry price actual"] == ""
        ):  # only process paper trades with no entry price info
            stock_code = row["Stock"]

            # For each stock, have to initiate a method with market params
            market = Market(row["Market"])

            entry_date_value = row["Entry date"]
            entry_date = arrow.get(entry_date_value, "DD/MM/YYYY").datetime.date()
            ohlc_daily, volume_daily = get_stock_data(
                f"{stock_code}{market.stock_suffix}", reporting_date_start
            )

            ohlc_daily["timestamp"] = ohlc_daily["timestamp"].dt.date
            ohlc_daily = ohlc_daily[
                ohlc_daily["timestamp"] >= entry_date
            ]  # only look from the entry date

            if len(ohlc_daily) > 0:
                open_price = round(ohlc_daily["open"].iloc[0], 3)  # take the first value (entry date)
                print(f"{stock_code} ({market.market_code}): {open_price}")
                update_row = (
                    index + 2
                )  # +2 to account for starting 0 and header
                gsheetsobj.sheet_update(
                    config["logging"]["gsheet_name"], tab_name, update_row,
                    config["logging"]["gsheet_actual_price_col"], open_price
                )
            else:
                print(
                    f"{stock_code} ({market.market_code}): no update needed yet"
                )


if __name__ == "__main__":
    print("Filling entry prices for paper trades...")
    fill_prices()
    print("Done")
