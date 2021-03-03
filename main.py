from libs.helpers import define_args, dates_diff, format_number
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
        print("Stocks list updated more than 5 days ago, please run the --update first")


def last_volume_5D_MA(volume_daily):
    volume_ma_20 = MA(volume_daily, 20, colname="volume")
    return volume_ma_20["ma20"].iloc[-1]


def met_conditions_bullish(ohlc_with_indicators_daily, volume_daily, ohlc_with_indicators_weekly):
    # Checks if price action meets conditions
    # Rules: MAs trending up, fast above slow, bullish TD count, volume spike
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
            and 1 <= ohlc_with_indicators_weekly["td_setup"].iloc[-1] < 8
    )

    # MA check
    ma_30 = MA(ohlc_with_indicators_daily, 30)
    ma_50 = MA(ohlc_with_indicators_daily, 50)
    ma_200 = MA(ohlc_with_indicators_daily, 200)
    ma_consensio = (
            ma_30["ma30"].iloc[-1] > ma_50["ma50"].iloc[-1] > ma_200["ma200"].iloc[-1]
    )

    # Volume MA and volume spike over the last 5 days
    volume_ma_20 = MA(volume_daily, 20, colname="volume")
    mergedDf = volume_daily.merge(volume_ma_20, left_index=True, right_index=True)
    mergedDf.dropna(inplace=True, how='any')
    mergedDf["volume_above_average"] = mergedDf['volume'].ge(mergedDf['ma20'])  # GE is greater or equal
    last_volume_to_ma = mergedDf["volume_above_average"][-5:].tolist()
    recent_volume_spike = True in last_volume_to_ma

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
        f"Closed higher: {daily_condition_close_higher} | Recent volume spike: {recent_volume_spike}"
    )

    return (
            daily_condition_td
            and weekly_condition_td
            and ma_consensio
            and ma_rising
            and not_overextended
            and daily_condition_close_higher
            and recent_volume_spike
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

        if met_conditions_bullish(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly
        ):
            print("- (v) meeting shortlisting conditions")
            volume_MA_5D = last_volume_5D_MA(volume_daily)
            shortlisted_stocks.append((stock.code, volume_MA_5D))
        else:
            print("- (x) not meeting shortlisting conditions")

    # Sort by volume descending
    sorted_stocks = sorted(shortlisted_stocks, key=lambda tup: tup[1], reverse=True)
    shortlist = [(val[0], val[1]) for val in sorted_stocks]

    print()
    print(f"All shortlisted stocks (sorted by 5-day moving average volume):")
    for stock in shortlist:
        print(f"- {stock[0]} ({format_number(stock[1])})")


if __name__ == "__main__":

    arguments = define_args()
    if arguments["update"]:
        print("Updating the ASX stocks list...")
        update_stocks()

    if arguments["scan"]:
        check_update_date()
        scan_stocks()
