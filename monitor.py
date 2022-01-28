# Monitors whether the open positions hit the exit criteria at some point
# Output W: wanted exit price, A: actual exit, D4: result if nothing on day 3 (so exit on 4), similar with D6
import libs.gsheetobj as gsheetsobj
from libs.stocktools import get_stock_data, get_stock_suffix, get_market_index_ticker
from libs.techanalysis import MA
from libs.settings import gsheet_name
import arrow
from datetime import timedelta
from libs.helpers import get_data_start_date
from libs.signal import market_bearish

reporting_date_start = get_data_start_date()

def get_first_true_idx(list):
    filtr = lambda x: x == True
    return [i for i, x in enumerate(list) if filtr(x)][0]


def check_market():
    exchange = "ASX"    # only supporting asx for now, easy to replicate for NASDAQ if needed
    market_ticker = get_market_index_ticker(exchange)
    market_ohlc_daily, market_volume_daily = get_stock_data(market_ticker, reporting_date_start)
    is_market_bearish, _ = market_bearish(market_ohlc_daily, market_volume_daily, output=True)
    if is_market_bearish:
        print("Overall market sentiment is bearish, exit all the open positions")
        exit(0)


def check_positions():
    alerted_positions = set()
    exchange = "ASX"    # only supporting asx for now, easy to replicate for NASDAQ if needed

    stock_suffix = get_stock_suffix(exchange)

    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")

    for index, row in ws.iterrows():
        if (
            row["Outcome"] == ""
        ):  # exclude the ones where we have results already, check if price falls below MA10

            ma10, mergedDf = None, None

            stock_code = row["Stock"]
            entry_date_value = row["Entry date"]

            try:
                entry_date = arrow.get(entry_date_value, "DD/MM/YYYY").datetime.date()
            except arrow.parser.ParserMatchError:
                print("Skipping blank entry date lines")
                continue  # continue with the next iteration in the for cycle

            ohlc_daily, volume_daily = get_stock_data(
                f"{stock_code}{stock_suffix}", reporting_date_start
            )
            ma10 = MA(ohlc_daily, 10)

            mergedDf = ohlc_daily.merge(ma10, left_index=True, right_index=True)
            mergedDf.dropna(inplace=True, how="any")

            mergedDf["close_below_ma"] = mergedDf["close"].lt(
                mergedDf["ma10"]
            )  # LT is lower than

            mergedDf["timestamp"] = mergedDf["timestamp"].dt.date
            mergedDf = mergedDf[
                mergedDf["timestamp"] >= entry_date
            ]  # only look from the entry date

            if len(mergedDf) == 0:
                continue  # skip to the next element if there is nothing yet

            # Also find the first date where price was lower than the entry date low
            entry_low = mergedDf["low"].values[0]
            mergedDf["close_below_entry_low"] = mergedDf["low"].lt(
                entry_low
            )  # LT is lower than

            alert = True in mergedDf["close_below_ma"].values
            if alert:
                hit_idx = get_first_true_idx(mergedDf["close_below_ma"].values)
                hit_date = mergedDf["timestamp"].values[hit_idx]

                if True in mergedDf["close_below_entry_low"].values:
                    hit_lower_than_low_idx = get_first_true_idx(mergedDf["close_below_entry_low"].values)
                    lower_than_low_date = mergedDf["timestamp"].values[hit_lower_than_low_idx]
                else:
                    lower_than_low_date = "-"

                exit_date = hit_date + timedelta(days=1)
                wanted_price = mergedDf["close"].values[hit_idx]
                ed_low = mergedDf["low"].values[0]
                ed_low_shifted = ed_low * 0.995  # minus 0.5% roughly
                ed_open = mergedDf["open"].values[0]
                entry_day_low_result = (ed_low_shifted - ed_open)/ed_open

                try:
                    opened_price = mergedDf["open"].values[hit_idx + 1]
                    # results for 4th day open and 6th day open
                    # for 'not moving for 3d / 5d' (first index is 0)
                    alerted_positions.add(
                        f"{stock_code} ({exchange}) [{entry_date} -> {exit_date}] "
                        f"wanted price: {round(wanted_price, 3)} | actual price: {round(opened_price, 3)}"
                    )
                    print(
                        f"{stock_code} ({exchange}) [{entry_date}]: alert"
                    )
                except IndexError:
                    alerted_positions.add(
                        f"{stock_code} ({exchange}) [{entry_date} -> {exit_date}]"
                    )
                    print(f"{stock_code} ({exchange}): alert (market pre-open)")

            else:
                print(
                    f"{stock_code} ({exchange}) [{entry_date}]: on track"
                )

    return alerted_positions


if __name__ == "__main__":
    print("Checking the market")
    check_market()

    print("Checking positions...")
    alerted_positions = check_positions()
    print()
    if len(alerted_positions) == 0:
        print("No alerts")
    else:
        print(f"Exit rules triggered for {len(alerted_positions)} stock(s):")
        for position in sorted(alerted_positions):
            print(f"- {position}")
