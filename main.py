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


def met_conditions_bullish(ohlc_with_indicators_daily, volume_daily, ohlc_with_indicators_weekly):
    # Checks if price action meets conditions. Rules: MAs trending up, fast above slow, bullish TD count
    daily_condition_close_higher = (  # closes higher
            ohlc_with_indicators_daily["close"].iloc[-1]
            > ohlc_with_indicators_daily["close"].iloc[-2]
    )
    daily_condition_td = (  # bullish TD count and more than 1st candle
            ohlc_with_indicators_daily["td_direction"].iloc[-1] == "green"
            and ohlc_with_indicators_daily["td_setup"].iloc[-1] > 1
    )
    weekly_condition_td = (  # bullish TD count and not exhausted
            ohlc_with_indicators_weekly["td_direction"].iloc[-1] == "green"
            and 1 < ohlc_with_indicators_weekly["td_setup"].iloc[-1] < 8
    )

    # MA check
    ma_30 = MA(ohlc_with_indicators_daily, 30)
    ma_50 = MA(ohlc_with_indicators_daily, 50)
    ma_200 = MA(ohlc_with_indicators_daily, 200)
    ma_consensio = (
            ma_30["ma30"].iloc[-1] > ma_50["ma50"].iloc[-1] > ma_200["ma200"].iloc[-1]
    )

    # All MA rising
    ma_rising = (
            (ma_30["ma30"].iloc[-1] > ma_30["ma30"].iloc[-5])
            and (ma_50["ma50"].iloc[-1] > ma_50["ma50"].iloc[-5])
            and (ma_200["ma200"].iloc[-1] > ma_200["ma200"].iloc[-5])
    )

    # Close for the last week is not more than 100% from the 4 weeks ago
    not_overextended = (
            ohlc_with_indicators_weekly["close"].iloc[-1]
            < 2 * ohlc_with_indicators_weekly["close"].iloc[-4]
    )

    print(
        f"-- TD conditions: {daily_condition_td} daily | {weekly_condition_td} weekly | "
        f"MA above: {ma_consensio} | MA rising: {ma_rising} | Not overextended: {not_overextended} | "
        f"Closed higher: {daily_condition_close_higher}"
    )

    return (
            daily_condition_td
            and weekly_condition_td
            and ma_consensio
            and ma_rising
            and not_overextended
            and daily_condition_close_higher
    )


def scan_stocks():
    shortlisted_stocks = []
    stocks = get_stocks(price_min=price_min, price_max=price_max)
    total_number = len(stocks)
    print(f"Scanning {total_number} stocks priced {price_min} from to {price_max}")

    for i, stock in enumerate(stocks):
        print(f"{stock.code} [{stock.name}] ({i + 1}/{len(stocks)})")
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

        # CONTINUE
        # Also need to get avg 5-d volume to sort by volume and to detect if there were spikes in volume over the last X days
        # TODO: add volume spike check to met_conditions_bullish
        if met_conditions_bullish(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly
        ):
            print("- (v) meeting bullish conditions")
            exit(0)
        # Check what's last close to swing low is
        # close_to_low = last_close_to_local_min(ohlc_with_indicators_daily)
        # stocks_meeting_condition_bullish.append((stock, close_to_low))
        else:
            print("- (x) not meeting bullish conditions")


if __name__ == "__main__":

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()
