# Monitors whether the open positions hit the exit criteria at some point
import libs.gsheetobj as gsheetsobj
from libs.stocktools import get_stock_data, get_stock_suffix
from libs.techanalysis import MA
from libs.settings import gsheet_name, trading_systems
import arrow


def check_positions():
    alerted_positions = set()
    for exchange in ["ASX", "NASDAQ"]:
        for system in trading_systems:

            stock_suffix = get_stock_suffix(exchange)

            ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange} - {system}")

            for index, row in ws.iterrows():
                if (
                    row["Outcome"] == ""
                ):  # exclude the ones where we have results already, check if price falls below MA10
                    stock_code = row["Stock"]
                    entry_date_value = row["Entry date"]
                    entry_date = arrow.get(entry_date_value, "DD/MM/YY").datetime.date()
                    ohlc_daily, volume_daily = get_stock_data(f"{stock_code}{stock_suffix}")
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

                    alert = True in mergedDf["close_below_ma"].values
                    if alert:
                        alerted_positions.add(f"{stock_code} ({exchange})")
                        print(f"{stock_code} ({exchange}) ({system}) [{entry_date}]: alert")
                    else:
                        print(f"{stock_code} ({exchange}) ({system}) [{entry_date}]: on track")

    return alerted_positions


if __name__ == "__main__":
    print("Checking positions...")
    alerted_positions = check_positions()
    print()
    if len(alerted_positions) == 0:
        print("No alerts")
    else:
        print("Exit rules triggered for:")
        for position in alerted_positions:
            print(f"- {position}")
