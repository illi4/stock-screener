# Simulates trading progress and results over time using a spreadsheet with various considerations
import libs.gsheetobj as gsheetsobj
from libs.settings import gsheet_name
import pandas as pd
from datetime import datetime, timedelta

# Settings
exchange = "ASX"
confidence_filter = [8, 9]
penny_filter = ["Y", "N"]
capital = 5000
commission = 5

# Variations to go through
simultaneous_positions = [2, 3, 4, 5]
variant_names = ["control", "test_a", "test_b", "test_c", "test_d"]
# exit_threshold_variants = ... # todo later
start_date = "2021-04-01"
end_date = "2021-05-01"

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


def p2f(s):
    stripped_s = s.strip("%")
    if stripped_s == "":
        return
    else:
        return float(stripped_s) / 100


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
    ws = ws.loc[ws["confidence"].isin(confidence_filter)]
    ws = ws.loc[ws["penny_stock"].isin(penny_filter)]

    ws["entry_date"] = pd.to_datetime(
        ws["entry_date"], format="%d/%m/%y", errors="coerce"
    )

    for variant_name in variant_names:
        ws[[f"time_in_trade_{variant_name}"]] = ws[
            [f"time_in_trade_{variant_name}"]
        ].apply(pd.to_numeric, errors="coerce")

        ws[f"{variant_name}_exit_date"] = pd.to_datetime(
            ws[f"{variant_name}_exit_date"], format="%d/%m/%y", errors="coerce"
        )
        ws[f"{variant_name}_result_%"] = ws[f"{variant_name}_result_%"].apply(p2f)

    return ws


if __name__ == "__main__":
    print("This solution will remain incomplete until there is an actual need for it")
    exit(0)

    print("Reading the values...")

    # This is working ok
    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")
    ws.columns = sheet_columns
    ws = convert_types(ws)

    # Dict to hold all the results
    results_dict = dict()

    # Iterating through days and simulations
    ### Iterate and check for entries / exits in a day depending on the variant ###
    current_variant = "control"
    current_capital = capital
    current_simultaneous_positions = 2
    positions_held = 0
    current_positions = set()

    # Stuff for stats
    winning_trades_number, losing_trades_number = 0, 0
    minimum_value = current_capital
    winning_trades, losing_trades = [], []

    # Starting
    print(f"Simulating {current_variant.capitalize()}")
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    current_date_dt = start_date_dt

    # Note (todo) to replace test D with the values same as in control if empty
    while current_date_dt < end_date_dt:

        current_date_dt = current_date_dt + timedelta(days=1)
        print(current_date_dt, current_positions)

        day_entries = ws.loc[ws["entry_date"] == current_date_dt.date]
        for key, elem in day_entries.iterrows():
            positions_held += 1
            current_positions.add(elem["stock"])
            print("Entry", elem["stock"], "| positions held", positions_held)

        day_exits = ws.loc[ws[f"{current_variant}_exit_date"] == current_date_dt.date]
        for key, elem in day_exits.iterrows():
            current_positions.remove(elem["stock"])
            positions_held -= 1
            result = elem[f"{current_variant}_result_%"]
            print(
                "Exit", elem["stock"], "| positions held", positions_held, "| ", result
            )
            current_capital = current_capital * (
                1 + elem[f"{current_variant}_result_%"] / current_simultaneous_positions
            )
            print("Capital:", current_capital)
