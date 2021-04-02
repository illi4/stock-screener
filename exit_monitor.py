import libs.gsheetobj as gsheetsobj
from libs.stocktools import (
    get_stock_data
)
from libs.techanalysis import MA
import arrow
import pandas as pd

stock_suffix = ".AX"  # test

print("Iterating through positions")

ws = gsheetsobj.sheet_to_df("Trading journal 2021", 'ASX')
for index, row in ws.iterrows():
    if row['Outcome'] == '':  # exclude the ones where we have results already, check if price falls below MA10
        stock_code = row['Stock']
        entry_date = arrow.get(row['Entry date'], 'DD/MM/YY').datetime.date()
        print(stock_code, entry_date)
        ohlc_daily, volume_daily = get_stock_data(f"{stock_code}{stock_suffix}")
        ma10 = MA(ohlc_daily, 10)

        mergedDf = ohlc_daily.merge(ma10, left_index=True, right_index=True)
        mergedDf.dropna(inplace=True, how="any")

        mergedDf["close_below_ma"] = mergedDf["close"].lt(
            mergedDf["ma10"]
        )  # LT is lower than

        mergedDf["timestamp"] = mergedDf["timestamp"].dt.date
        mergedDf = mergedDf[mergedDf["timestamp"] >= entry_date]

        print("Result", True in mergedDf["close_below_ma"].values)