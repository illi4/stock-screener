# Fills in the missing prices for paper entries where applicable using daily data
# Intraday is not implemented because of the higher cost for intraday data

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import libs.gsheetobj as gsheetsobj
from libs.helpers import get_data_start_date
import arrow
from libs.techanalysis import td_indicators, MA, fisher_distance, coppock_curve
from libs.stocktools import (
    get_stock_data,
    ohlc_daily_to_weekly,
    Market
)

from libs.read_settings import read_config
config = read_config()

from time import sleep

reporting_date_start = get_data_start_date()
sheet_name = config["logging"]["gsheet_name"]
tab_name = config["logging"]["gsheet_tab_name"]

def fill_prices():

    ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)
    price_column_name = 'Entry price allocation 1'

    for index, row in ws.iterrows():
        if (
            row["Trade type"] == "paper"
            and row[price_column_name] == ""
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

                # gsheetsobj.sheet_update(
                #     config["logging"]["gsheet_name"], tab_name, update_row,
                #     config["logging"]["gsheet_actual_price_col"], open_price
                # )
                gsheet_name = config["logging"]["gsheet_name"]
                gsheetsobj.sheet_update_by_column_name(gsheet_name, tab_name, update_row,
                                                       price_column_name, open_price)

            else:
                print(
                    f"{stock_code} ({market.market_code}): no update needed yet"
                )

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


def calculate_metric(func, df):
    """Calculate and round a metric using a given function and DataFrame."""
    return round(func(df).iloc[-1].values[0], 2)


def backfill_metrics():

    ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)

    for index, row in ws.iterrows():
        if (
            row["fisherDaily"] == "" or
            row["fisherWeekly"] == "" or
            row["coppockDaily"] == "" or
            row["coppockWeekly"] == ""
        ):  # only process trades with no metric values
            stock_code = row["Stock"]

            # For each stock, have to initiate a method with market params
            market = Market(row["Market"])

            entry_date_value = row["Entry date"]
            entry_date = arrow.get(entry_date_value, "DD/MM/YYYY").datetime.date()
            ohlc_daily, volume_daily = get_stock_data(
                f"{stock_code}{market.stock_suffix}", reporting_date_start
            )

            ohlc_daily["timestamp"] = ohlc_daily["timestamp"].dt.date

            # It should stop at the entry date, works ok
            ohlc_daily = ohlc_daily[
                ohlc_daily["timestamp"] < entry_date
            ]
            ohlc_daily["timestamp"] = pd.to_datetime(ohlc_daily["timestamp"], errors='coerce')

            # Get dataframes
            (
                ohlc_with_indicators_daily,
                ohlc_with_indicators_weekly,
            ) = generate_indicators_daily_weekly(ohlc_daily)

            # Calculate extra metrics
            fisher_daily = calculate_metric(fisher_distance, ohlc_with_indicators_daily)
            fisher_weekly = calculate_metric(fisher_distance, ohlc_with_indicators_weekly)
            coppock_daily = calculate_metric(coppock_curve, ohlc_with_indicators_daily)
            coppock_weekly = calculate_metric(coppock_curve, ohlc_with_indicators_weekly)

            # Backfill
            if len(ohlc_daily) > 0:

                print(f"{stock_code} ({market.market_code}): {fisher_daily}|{fisher_weekly}|{coppock_daily}|{coppock_weekly}")
                update_row = (
                    index + 2
                )  # +2 to account for starting 0 and header

                # Update everything
                columns = {
                    'fisherDaily': fisher_daily,
                    'fisherWeekly': fisher_weekly,
                    'coppockDaily': coppock_daily,
                    'coppockWeekly': coppock_weekly
                }
                gsheet_name = config["logging"]["gsheet_name"]

                for column, value in columns.items():
                    gsheetsobj.sheet_update_by_column_name(gsheet_name, tab_name, update_row, column, value)
                    sleep(2)  # Respect request quota

            else:
                print(
                    f"{stock_code} ({market.market_code}): no update possible"
                )


if __name__ == "__main__":
    print("Filling entry prices for paper trades...")
    fill_prices()

    # Backfill metrics
    # Not currently used, was used when doing R&D
    # print("Backfilling indicator values")
    # backfill_metrics()

    print("Done")
