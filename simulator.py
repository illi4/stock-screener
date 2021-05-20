# Simulates trading progress and results over time using a spreadsheet with various considerations
import libs.gsheetobj as gsheetsobj
from libs.settings import gsheet_name
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Settings
exchange = "ASX"
confidence_filter = [8, 9]
penny_filter = ["Y", "N"]
capital = 5000
commission = 10

# Variations to go through
simultaneous_positions = [2, 3, 4, 5]
variant_names = ["control", "test_a", "test_b", "test_c", "test_d"]
# exit_threshold_variants = ... # todo later
# take_profit_levels = [0.25, 0.5, 0.9] # to use outside of control, test a, etc.
start_date = "2021-04-01"
end_date = "2021-05-20"

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
    "test_e_result_%",
    "max_level_reached",
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


def calculate_max_drawdown(capital_change_values):
    """
    Function to calculate max drawdown

    :param capital_change_values: list of capital values changed over time
    :return: maximum drawdown in decimal (e.g. 0.65 means 65%)
    """
    maximums = np.maximum.accumulate(capital_change_values)
    drawdowns = 1 - capital_change_values / maximums
    try:  # does not work when there are no drawdowns
        j = np.argmax(
            np.maximum.accumulate(capital_change_values) - capital_change_values
        )  # end of the periods
        i = np.argmax(capital_change_values[:j])  # start of period
        diff = j - i  # difference in periods
    except ValueError:
        diff = 0
    return np.max(drawdowns)


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
    ws["max_level_reached"] = ws["max_level_reached"].apply(p2f)

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

    print("reading the values...")

    # This is working ok
    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")
    ws.columns = sheet_columns
    ws = convert_types(ws)

    # Dict to hold all the results
    results_dict = dict()

    # Iterating through days and simulations
    ### Iterate and check for entries / exits in a day depending on the variant ###
    for current_variant in variant_names:
        for current_simultaneous_positions in simultaneous_positions:
            # current_variant = "control" # for testing
            # current_simultaneous_positions = 4 # for testing
            current_capital = capital
            positions_held = 0
            current_positions = set()
            capital_values = []
            capital_values.append(current_capital)

            # Stuff for stats
            winning_trades_number, losing_trades_number = 0, 0
            minimum_value = current_capital
            winning_trades, losing_trades = [], []
            worst_trade_adjusted, best_trade_adjusted = 0, 0

            # Starting for a variant
            print(f"simulating {current_variant.lower()}")
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            current_date_dt = start_date_dt

            # Balance for the start of the period which will then be updated
            balances = dict()
            balances[start_date_dt.strftime("%d/%m/%Y")] = current_capital

            # Iterating over days
            while current_date_dt < end_date_dt:

                previous_date_month = current_date_dt.strftime("%m")
                current_date_dt = current_date_dt + timedelta(days=1)
                current_date_month = current_date_dt.strftime("%m")

                if previous_date_month != current_date_month:
                    balances[current_date_dt.strftime("%d/%m/%Y")] = current_capital

                print(current_date_dt, "| positions: ", current_positions)
                day_entries = ws.loc[ws["entry_date"] == current_date_dt]
                for key, elem in day_entries.iterrows():

                    if len(current_positions) + 1 > current_simultaneous_positions:
                        print(f"max possible positions held, skipping {elem['stock']}")
                    else:
                        positions_held += 1
                        current_positions.add(elem["stock"])
                        print("-> entry", elem["stock"], "| positions held", positions_held)
                        print(f"accounting for the trade price: ${commission}")
                        current_capital -= commission

                day_exits = ws.loc[ws[f"{current_variant}_exit_date"] == current_date_dt]
                for key, elem in day_exits.iterrows():
                    if elem["stock"] in current_positions:
                        current_positions.remove(elem["stock"])
                        positions_held -= 1
                        result = elem[f"{current_variant}_result_%"]

                        if result >= 0:
                            winning_trades_number += 1
                            winning_trades.append(result)
                            if (best_trade_adjusted is None) or (
                                result / current_simultaneous_positions
                                > best_trade_adjusted
                            ):
                                best_trade_adjusted = (
                                    result / current_simultaneous_positions
                                )
                        elif result < 0:
                            losing_trades_number += 1
                            losing_trades.append(result)
                            if (worst_trade_adjusted is None) or (
                                result / current_simultaneous_positions
                                < worst_trade_adjusted
                            ):
                                worst_trade_adjusted = (
                                    result / current_simultaneous_positions
                                )

                        print(
                            f'-> exit {elem["stock"]} | result: {result:.2%} | positions held {positions_held}'
                        )
                        current_capital = current_capital * (
                            1
                            + elem[f"{current_variant}_result_%"]
                            / current_simultaneous_positions
                        )
                        print(f"accounting for the trade price: ${commission}")
                        current_capital -= commission
                        print(f"balance: ${current_capital}")
                        capital_values.append(current_capital)

            balances[
                current_date_dt.strftime("%d/%m/%Y")
            ] = current_capital  # for the end date
            print("result:", balances)

            # other metrics
            growth = (current_capital - capital) / capital
            win_rate = (winning_trades_number) / (
                winning_trades_number + losing_trades_number
            )
            print(f"capital growth/loss: {growth:.2%}")
            print(
                f"win rate: {win_rate:.2%} | winning_trades: {winning_trades_number} | losing trades: {losing_trades_number}"
            )
            print(
                f"best trade (adjusted for sizing) {best_trade_adjusted:.2%} | worst trade (adjusted for sizing) {worst_trade_adjusted:.2%}"
            )
            max_drawdown = calculate_max_drawdown(capital_values)
            print(f"max_drawdown: {max_drawdown}:.2%")

            # saving the result
            result_current_dict = dict(
                growth=growth * 100,
                win_rate=win_rate * 100,
                winning_trades_number=winning_trades_number,
                losing_trades_number=losing_trades_number,
                best_trade_adjusted=best_trade_adjusted * 100,
                worst_trade_adjusted=worst_trade_adjusted * 100,
                max_drawdown=max_drawdown * 100,
                simultaneous_positions=current_simultaneous_positions,
                variant_group=current_variant
            )
            results_dict[f"{current_variant}_{current_simultaneous_positions}pos"] = result_current_dict

    #### Iterations finished


    ##### More complex implementation - various options on threshold levels

    ##### Finalisation
    # write the output to a dataframe and a spreadsheet
    resulting_dataframes = []
    for k, v in results_dict.items():
        print(k, v)
        values_current = v.copy()
        values_current["variant"] = k
        resulting_dataframes.append(
            pd.DataFrame.from_records(values_current, index=[0])
        )

    final_result = pd.concat(df for df in resulting_dataframes)
    final_result = final_result[[
        "variant",
        "simultaneous_positions",
        "variant_group",
        "best_trade_adjusted",
        "growth",
        "losing_trades_number",
        "max_drawdown",
        "win_rate",
        "winning_trades_number",
        "worst_trade_adjusted",
    ]]

    print(final_result)

    # save to csv
    final_result.to_csv("simulator_result.csv", index=False)