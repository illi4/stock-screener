# Monitors whether the open positions hit the exit criteria at some point
# Output W: wanted exit price, A: actual exit, D4: result if nothing on day 3 (so exit on 4), similar with D6
import libs.gsheetobj as gsheetsobj
from libs.stocktools import get_stock_data, get_stock_suffix
from libs.techanalysis import MA
from libs.settings import gsheet_name, trading_systems
import arrow
from datetime import timedelta


def get_first_true_idx(list):
    filtr = lambda x: x == True
    return [i for i, x in enumerate(list) if filtr(x)][0]


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
                    ohlc_daily, volume_daily = get_stock_data(
                        f"{stock_code}{stock_suffix}"
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

                    alert = True in mergedDf["close_below_ma"].values
                    if alert:
                        hit_idx = get_first_true_idx(mergedDf["close_below_ma"].values)
                        hit_date = mergedDf["timestamp"].values[hit_idx]
                        exit_date = hit_date + timedelta(days=1)
                        wanted_price = mergedDf["close"].values[hit_idx]

                        try:
                            opened_price = mergedDf["open"].values[hit_idx + 1]
                            # results for 4th day open and 6th day open
                            # for 'not moving for 3d / 5d' (first index is 0)

                            try:
                                result_d_4 = (
                                    mergedDf["open"].values[3]
                                    - mergedDf["open"].values[0]
                                ) / (mergedDf["open"].values[0])
                                date_d4 = mergedDf["timestamp"].values[3]
                            except IndexError:
                                result_d_4 = (
                                    mergedDf["open"].values[-1]
                                    - mergedDf["open"].values[0]
                                ) / (mergedDf["open"].values[0])
                                date_d4 = mergedDf["timestamp"].values[-1]

                            try:
                                result_d_6 = (
                                    mergedDf["open"].values[5]
                                    - mergedDf["open"].values[0]
                                ) / (mergedDf["open"].values[0])
                                date_d6 = mergedDf["timestamp"].values[5]
                            except IndexError:
                                result_d_6 = (
                                    mergedDf["open"].values[-1]
                                    - mergedDf["open"].values[0]
                                ) / (mergedDf["open"].values[0])
                                date_d6 = mergedDf["timestamp"].values[-1]

                            alerted_positions.add(
                                f"{stock_code} ({exchange}) [{entry_date} -> {exit_date}] "
                                f"W {round(wanted_price, 3)} A {round(opened_price, 3)} D4 ({date_d4}) {result_d_4:.2%} "
                                f"D6 ({date_d6}) {result_d_6:.2%}"
                            )
                            print(
                                f"{stock_code} ({exchange}) ({system}) [{entry_date}]: alert"
                            )
                        except IndexError:
                            print(f"Too early to add for {stock_code}")

                    else:
                        print(
                            f"{stock_code} ({exchange}) ({system}) [{entry_date}]: on track"
                        )

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
