# Simulates trading progress and results over time using a spreadsheet with various considerations
import libs.gsheetobj as gsheetsobj
from libs.settings import gsheet_name
import pandas as pd

# Settings
exchange = "ASX"
confidence_filter = [8, 9]
variant_names = ["control", "test_a", "test_b", "test_c", "test_d"]
capital = 5000
commission = 5

# Variations to go through
simultaneous_positions = [2, 3, 4, 5]

# Sheet columns for the Gsheet
sheet_columns = [
    "stock",
    "trade_type",
    "entry_date",
    "entry_price_planned",
    "entry_price_actual",
    "confidence",
    "penny_stock",
    "control_exit_date",
    "exit_price_planned",
    "exit_price_actual",
    "outcome",
    "control_result_%",
    "test_a_exit_date",
    "test_a_result_%",
    "test_b_exit_date",
    "test_b_result_%",
    "test_c_exit_date",
    "test_c_result_%",
    "test_d_price",
    "test_d_exit_date",
    "test_d_result_%",
    "comments",
    "time_in_trade_control",
    "time_in_trade_test_a",
    "time_in_trade_test_b",
    "time_in_trade_test_c",
    "time_in_trade_test_d",
]


def convert_types(ws):
    # Convert types
    num_cols = [
        "confidence",
        "entry_price_planned",
        "entry_price_actual",
        "exit_price_planned",
        "exit_price_actual",
    ]
    ws[num_cols] = ws[num_cols].apply(pd.to_numeric, errors="coerce")
    dt_cols = ["entry_date"]
    ws[dt_cols] = ws[dt_cols].apply(pd.to_datetime, errors="coerce")

    for variant_name in variant_names:
        ws[[f"{variant_name}_result_%", f"time_in_trade_{variant_name}"]] = ws[
            [f"{variant_name}_result_%", f"time_in_trade_{variant_name}"]
        ].apply(pd.to_numeric, errors="coerce")
        ws[[f"{variant_name}_exit_date"]] = ws[[f"{variant_name}_exit_date"]].apply(pd.to_datetime, errors="coerce")

    ws = ws.loc[ws["confidence"].isin(confidence_filter)]

    return ws


if __name__ == "__main__":
    print("Reading the values...")

    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")
    ws.columns = sheet_columns

    ws = convert_types(ws)

    print(ws.head())
    print(ws.dtypes)