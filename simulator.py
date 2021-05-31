# Simulates trading progress and results over time using a spreadsheet with various considerations
# Will save the result to simulator_result.csv
import libs.gsheetobj as gsheetsobj
from libs.settings import gsheet_name
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from libs.stocktools import get_stock_suffix, get_stock_data
import argparse
parser = argparse.ArgumentParser()

# Settings
exchange = "ASX"
confidence_filter = [8, 9]
penny_filter = ["Y", "N"]
capital = 5000
commission = 10

# Variations to go through
simultaneous_positions = [2, 3, 4, 5]
variant_names = ["control", "test_a", "test_b", "test_c", "test_d", "test_e"]
start_date = "2021-04-01"
end_date = "2021-06-02"

# Take profit level variations
# Would be used iterating over control with simultaneous_positions variations too
take_profit_variants = {
    #'repeated_to_control':[0.25, 0.45, 0.9],  # this is just to check that I get the same result as in control if testing is needed
    "tp_a": [0.3, 0.5, 0.9],
    "tp_b": [0.5, 1],
    "tp_c": [0.15, 0.5, 0.9],
    "tp_d": [0.5, 1, 1.5],
    "tp_e": [0.25, 0.45, 0.9, 1.45],
}

# Sheet columns for the Gsheet
sheet_columns = [
    "stock",
    "trade_type",
    "entry_date",
    "entry_price_planned",
    "entry_price_actual",
    "confidence",
    "penny_stock",
    "higher_open",
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
    "test_e_exit_date",
    "test_e_result_%",
    "max_level_reached",
    "comments",
    "time_in_trade_control",
    "time_in_trade_test_a",
    "time_in_trade_test_b",
    "time_in_trade_test_c",
    "time_in_trade_test_d",
    "time_in_trade_test_e",
]


def define_args():
    # Take profit levels variation is only supported for the control group, thus the modes are different
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        help="Mode to run the simulation in (main|tp). Tp mode is only applied to control.",
        choices=["main", "tp"],
    )

    args = parser.parse_args()
    arguments = vars(args)
    arguments["mode"] = arguments["mode"].lower()

    return arguments


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


class simulation:
    def __init__(self, capital):
        self.current_capital = capital
        self.minimum_value = capital
        self.positions_held = 0
        self.current_positions = set()
        self.capital_values = []
        self.winning_trades_number, self.losing_trades_number = 0, 0
        self.winning_trades, self.losing_trades = [], []
        self.worst_trade_adjusted, self.best_trade_adjusted = 0, 0
        self.balances = dict()
        self.capital_values.append(self.current_capital)
        self.growth, self.win_rate, self.max_drawdown, self.mom_growth = None, None, None, None
        # For thresholds
        self.left_of_initial_entries = dict()  # list of initial entries
        self.thresholds_reached = dict()  # for the thresholds reached
        self.entry_prices = dict()  # for the entry prices

    def snapshot_balance(self, current_date_dt):
        self.balances[
            current_date_dt.strftime("%d/%m/%Y")
        ] = self.current_capital  # for the end date
        print("balances:", self.balances)

    def remove_stock_traces(self, stock):
        sim.left_of_initial_entries.pop(stock, None)
        sim.thresholds_reached.pop(stock, None)
        sim.entry_prices.pop(stock, None)
        sim.thresholds_reached[stock] = []


def add_entry_no_profit_thresholds(sim, stock):
    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock}")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        print(
            "-> entry",
            stock,
            "| positions held",
            sim.positions_held,
        )
        print(f"accounting for the trade price: ${commission}")
        sim.current_capital -= commission


def add_exit_no_profit_thresholds(sim, stock, result):
    if stock in sim.current_positions:
        sim.current_positions.remove(stock)
        sim.positions_held -= 1

        if result >= 0:
            sim.winning_trades_number += 1
            sim.winning_trades.append(result)
            if (sim.best_trade_adjusted is None) or (
                result / current_simultaneous_positions > sim.best_trade_adjusted
            ):
                sim.best_trade_adjusted = result / current_simultaneous_positions
        elif result < 0:
            sim.losing_trades_number += 1
            sim.losing_trades.append(result)
            if (sim.worst_trade_adjusted is None) or (
                result / current_simultaneous_positions < sim.worst_trade_adjusted
            ):
                sim.worst_trade_adjusted = result / current_simultaneous_positions

        print(
            f"-> exit {stock} | result: {result:.2%} | positions held {sim.positions_held}"
        )
        sim.current_capital = sim.current_capital * (
            1 + elem[f"{current_variant}_result_%"] / current_simultaneous_positions
        )
        print(f"accounting for the trade price: ${commission}")
        sim.current_capital -= commission
        print(f"balance: ${sim.current_capital}")
        sim.capital_values.append(sim.current_capital)


def mean_mom_growth(balances):
    balances = np.array(balances)
    diff_list = np.diff(balances)
    balances_shifted = balances[:-1]
    mom_growth = diff_list/balances_shifted
    return mom_growth.mean()


def calculate_metrics(sim, capital):
    sim.growth = (sim.current_capital - capital) / capital
    sim.win_rate = (sim.winning_trades_number) / (
        sim.winning_trades_number + sim.losing_trades_number
    )
    sim.max_drawdown = calculate_max_drawdown(sim.capital_values)
    balances = [v for k, v in sim.balances.items()]
    sim.mom_growth = mean_mom_growth(balances)


def print_metrics(sim):
    print(f"capital growth/loss: {sim.growth:.2%}")
    print(
        f"win rate: {sim.win_rate:.2%} | winning_trades: {sim.winning_trades_number} | losing trades: {sim.losing_trades_number}"
    )
    print(
        f"best trade (adjusted for sizing) {sim.best_trade_adjusted:.2%} | worst trade (adjusted for sizing) {sim.worst_trade_adjusted:.2%}"
    )
    print(f"max_drawdown: {sim.max_drawdown}:.2%")


def update_results_dict(
    results_dict, sim, current_simultaneous_positions, current_variant, extra_suffix=""
):
    result_current_dict = dict(
        growth=sim.growth * 100,
        win_rate=sim.win_rate * 100,
        winning_trades_number=sim.winning_trades_number,
        losing_trades_number=sim.losing_trades_number,
        best_trade_adjusted=sim.best_trade_adjusted * 100,
        worst_trade_adjusted=sim.worst_trade_adjusted * 100,
        max_drawdown=sim.max_drawdown * 100,
        avg_mom_growth=sim.mom_growth * 100,
        simultaneous_positions=current_simultaneous_positions,
        variant_group=current_variant,
    )
    results_dict[
        f"{current_variant}_{current_simultaneous_positions}pos{extra_suffix}"
    ] = result_current_dict
    return results_dict


def add_entry_with_profit_thresholds(sim, stock, entry_price_actual):
    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock}")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        print(
            "-> entry",
            stock,
            "| positions held",
            sim.positions_held,
        )
        print(f"-> entry price: {entry_price_actual}")
        print(f"accounting for the trade price: ${commission}")
        sim.current_capital -= commission

        # on the entry, we have the full position
        sim.left_of_initial_entries[stock] = 1
        sim.entry_prices[stock] = entry_price_actual

        # also, on the entry we initiate the dict of thresholds hit for the item
        # they will then be populated with like (0.25, ...)
        sim.thresholds_reached[
            stock
        ] = set()  # appropriate as each would only be there once


def thresholds_check(sim, stock_prices, current_date_dt):
    for position in sim.current_positions:
        current_df = stock_prices[position][0]
        curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
        if not curr_row.empty:
            print(
                f"Current high for {position} {curr_row['high'].iloc[0]} | entry: {sim.entry_prices[position]}"
            )
            for each_threshold in current_tp_variant:
                if curr_row["high"].iloc[0] > sim.entry_prices[position] * (
                    1 + each_threshold
                ):
                    sim.thresholds_reached[position].add(each_threshold)
                    print("-- reached", each_threshold)


def add_exit_with_profit_thresholds(
    sim, stock, entry_price_actual, exit_price_actual, control_result_percent
):
    if stock in sim.current_positions:
        position = stock
        sim.current_positions.remove(stock)
        sim.positions_held -= 1

        # check what thresholds were reached. use entry and exit price and thresholds reached
        print(sim.thresholds_reached[position])
        number_thresholds_total = len(current_tp_variant)
        number_thresholds_reached = len(sim.thresholds_reached[position])
        divisor = number_thresholds_total + 1
        portion_not_from_thresholds = divisor - number_thresholds_reached

        absolute_price_result = (
            exit_price_actual - entry_price_actual
        ) / entry_price_actual
        result = absolute_price_result * portion_not_from_thresholds / divisor
        print(f"Price change result for {position}: {result}")
        print(
            f"Thresholds reached ({position}): {sim.thresholds_reached[position]}: {number_thresholds_reached} of {number_thresholds_total}"
        )

        for threshold_reached in sim.thresholds_reached[position]:
            result += threshold_reached / divisor

        print(f"Result ({position}) accounting for thresholds reached: {result}")

        if result >= 0:
            sim.winning_trades_number += 1
            sim.winning_trades.append(result)
            if (sim.best_trade_adjusted is None) or (
                result / current_simultaneous_positions > sim.best_trade_adjusted
            ):
                sim.best_trade_adjusted = result / current_simultaneous_positions
                print(f"best_trade_adjusted is now {sim.best_trade_adjusted}")
        elif result < 0:
            sim.losing_trades_number += 1
            sim.losing_trades.append(result)
            if (sim.worst_trade_adjusted is None) or (
                result / current_simultaneous_positions < sim.worst_trade_adjusted
            ):
                sim.worst_trade_adjusted = result / current_simultaneous_positions

        print(
            f"-> exit {stock} | result: {result:.2%} | positions held {sim.positions_held}"
        )

        sim.current_capital = sim.current_capital * (
            1 + control_result_percent / current_simultaneous_positions
        )
        print(f"accounting for the trade price: ${commission}")
        sim.current_capital -= commission
        print(f"balance: ${sim.current_capital}")
        sim.capital_values.append(sim.current_capital)

        # Delete from the partial positions left, prices, thresholds for the element
        sim.remove_stock_traces(elem["stock"])


if __name__ == "__main__":

    arguments = define_args()

    print("reading the values...")

    # This is working ok
    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")
    ws.columns = sheet_columns
    ws = convert_types(ws)

    # Dict to hold all the results
    results_dict = dict()

    # > Iterating through days and variants for the fixed TP levels per the control & spreadsheet
    if arguments["mode"] == "main":
        for current_variant in variant_names:
            for current_simultaneous_positions in simultaneous_positions:

                # Initiate the simulation object
                sim = simulation(capital)

                # Starting for a variant
                print(f"simulating {current_variant.lower()}")
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_dt = start_date_dt

                # Balance for the start of the period which will then be updated
                sim.balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

                # Iterating over days
                while current_date_dt < end_date_dt:

                    previous_date_month = current_date_dt.strftime("%m")
                    current_date_dt = current_date_dt + timedelta(days=1)
                    current_date_month = current_date_dt.strftime("%m")

                    if previous_date_month != current_date_month:
                        sim.balances[
                            current_date_dt.strftime("%d/%m/%Y")
                        ] = sim.current_capital

                    print(current_date_dt, "| positions: ", sim.current_positions)

                    # Entries
                    day_entries = ws.loc[ws["entry_date"] == current_date_dt]
                    for key, elem in day_entries.iterrows():
                        add_entry_no_profit_thresholds(sim, elem["stock"])

                    # Exits
                    day_exits = ws.loc[
                        ws[f"{current_variant}_exit_date"] == current_date_dt
                    ]
                    for key, elem in day_exits.iterrows():
                        add_exit_no_profit_thresholds(
                            sim, elem["stock"], elem[f"{current_variant}_result_%"]
                        )

                # Add the final balance at the end of the date
                # sim.snapshot_balance(current_date_dt) # nope, makes mom calc convoluted

                # Calculate metrics and print the results
                calculate_metrics(sim, capital)
                print_metrics(sim)

                # Saving the result in the overall dictionary
                results_dict = update_results_dict(
                    results_dict,
                    sim,
                    current_simultaneous_positions,
                    current_variant,
                )

    # < Finished iterating

    # > Iterating through days and take profit variants for the dynamic TP levels
    # Only supported for control but allows to make some conclusions too
    if arguments["mode"] == "tp":
        current_variant = "control"
        stock_names = [item.stock for key, item in ws.iterrows()]
        stock_prices = dict()
        suffix = get_stock_suffix(exchange)
        for stock in stock_names:
            print(f"getting stock data for {stock}{suffix}")
            stock_info = get_stock_data(f"{stock}{suffix}")
            stock_prices[stock] = stock_info

        for current_tp_variant_name, current_tp_variant in take_profit_variants.items():
            for current_simultaneous_positions in simultaneous_positions:

                # Initiate the simulation object
                sim = simulation(capital)

                # Starting for a variant
                print(f"simulating control with TP levels {current_tp_variant}")
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_dt = start_date_dt

                # Balance for the start of the period which will then be updated
                balances = dict()
                balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

                # Iterating over days
                while current_date_dt < end_date_dt:

                    previous_date_month = current_date_dt.strftime("%m")
                    current_date_dt = current_date_dt + timedelta(days=1)
                    current_date_month = current_date_dt.strftime("%m")

                    if previous_date_month != current_date_month:
                        balances[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

                    print(
                        current_date_dt,
                        "| positions: ",
                        sim.current_positions,
                        "| left of entries:",
                        sim.left_of_initial_entries,
                    )

                    # Entries
                    day_entries = ws.loc[ws["entry_date"] == current_date_dt]
                    for key, elem in day_entries.iterrows():
                        add_entry_with_profit_thresholds(
                            sim, elem["stock"], elem["entry_price_actual"]
                        )

                    # For each day, need to check the current positions and whether the position reached a threshold
                    thresholds_check(sim, stock_prices, current_date_dt)

                    # Exits
                    day_exits = ws.loc[ws[f"control_exit_date"] == current_date_dt]
                    for (
                        key,
                        elem,
                    ) in day_exits.iterrows():
                        add_exit_with_profit_thresholds(
                            sim,
                            elem["stock"],
                            elem[f"entry_price_actual"],
                            elem[f"exit_price_actual"],
                            elem[f"control_result_%"],
                        )

                # Add the final balance at the end of the date
                # sim.snapshot_balance(current_date_dt) # nope, makes mom calc convoluted

                # Calculate metrics and print the results
                calculate_metrics(sim, capital)
                print_metrics(sim)

                # Saving the result in the overall dictionary
                results_dict = update_results_dict(
                    results_dict,
                    sim,
                    current_simultaneous_positions,
                    current_variant,
                    extra_suffix=f"_tp{current_tp_variant_name}",
                )

    # < Finished iterating

    # Finalisation
    # Write the output to a dataframe and a spreadsheet
    resulting_dataframes = []
    for k, v in results_dict.items():
        print(k, v)
        values_current = v.copy()
        values_current["variant"] = k
        resulting_dataframes.append(
            pd.DataFrame.from_records(values_current, index=[0])
        )

    final_result = pd.concat(df for df in resulting_dataframes)
    final_result = final_result[
        [
            "variant",
            "simultaneous_positions",
            "variant_group",
            "best_trade_adjusted",
            "growth",
            "losing_trades_number",
            "max_drawdown",
            "win_rate",
            "avg_mom_growth",
            "winning_trades_number",
            "worst_trade_adjusted"
        ]
    ]

    # just an info bit
    if arguments["mode"] == "tp":
        print(f"take profit variants tested: {take_profit_variants}")

    # save to csv
    final_result.to_csv("simulator_result.csv", index=False)
    print("results saved to simulator_result.csv")