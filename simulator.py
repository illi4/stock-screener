# Simulates trading progress and results over time using a spreadsheet with various considerations
# Will save the result to simulator_result.csv
# Use example: python simulator.py -mode=main -exchange=nasdaq -start=2021-02-01 -end=2021-07-01

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

import csv
import libs.gsheetobj as gsheetsobj
from libs.signal import red_day_on_volume
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from libs.stocktools import get_stock_suffix, get_stock_data
import argparse
from itertools import groupby

parser = argparse.ArgumentParser()
import matplotlib.pyplot as plt

from libs.read_settings import read_config
config = read_config()

pd.set_option("display.max_columns", None)

# Settings (default)
# higher_or_equal_open_filter, higher_strictly_open_filter, and red_entry_day_exit
# are returned from process_filter_args()
capital = 100000
commission = 0  # this is brokerage (per entry / per exit)

# Variations to go through
simultaneous_positions = [4] # [3, 4, 5]

# Quick and dirty check of the 'failsafe level' hypothesis
failsafe_trigger_level = 0.15
failsafe_exit_level = 0.05

# Take profit level variations
# Would be used iterating over control with simultaneous_positions variations too
take_profit_variants = {
    #"_repeated_to_control": [0.25, 0.45, 0.9], # preserved this for historical purposes
    #"tp_b": [0.5, 1],
    #"tp_c": [0.15, 0.5, 0.9, 1.75],
    "tp_d": [0.5, 1, 1.5],
    #"tp_e": [0.25, 0.45, 0.9, 1.45],
    #"tp_g": [0.25, 0.9, 1.45, 1.75],
    # "tp_h": [1.45, 1.75, 1.95],
    # "tp_k1": [0.45, 1.75, 1.95],
    # "tp_l1": [0.45, 1.45, 1.75, 1.95],
    # "tp_x": [0.1, 1.45, 1.75],
    # "tp_y": [0.1, 1.75, 1.95]
}


def define_args():
    # Take profit levels variation is only supported for the control group, thus the modes are different
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        help="Mode to run the simulation in (main|tp). Main mode means taking profit as per the spreadsheet setup.",
        choices=["main", "tp"],
    )
    parser.add_argument(
        "-exchange",
        type=str,
        required=True,
        help="Exchange (asx|nasdaq)",
        choices=["asx", "nasdaq"],
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the latest simulation"
    )
    parser.add_argument(
        "--failsafe", action="store_true", help="Activate the failsafe approach"
    )

    parser.add_argument(
        "--show_monthly", action="store_true", help="Show MoM capital value (only in main mode)"
    )

    # Adding the dates
    parser.add_argument(
        "-start",
        type=str,
        required=True,
        help="Start date to run for (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "-end",
        type=str,
        required=True,
        help="End date to run for (YYYY-MM-DD format)",
    )

    # Arguments to overwrite default settings for filtering
    parser.add_argument(
        "--red_day_exit",
        action="store_true",
        help="Exit when entry day is red (in tp mode only)",
    )

    args = parser.parse_args()
    arguments = vars(args)
    arguments["mode"] = arguments["mode"].lower()
    arguments["exchange"] = arguments["exchange"].upper()
    if not arguments["plot"]:
        arguments["plot"] = False
    if not arguments["failsafe"]:
        arguments["failsafe"] = False
    if not arguments["show_monthly"]:
        arguments["show_monthly"] = False

    # Check for consistency
    if arguments["mode"] == 'tp' and arguments["show_monthly"]:
        print('Error: show_monthly option is only supported in the main mode')

    return arguments


def process_filter_args():
    red_entry_day_exit = True if arguments["red_day_exit"] else False
    failsafe = True if arguments["failsafe"] else False
    return red_entry_day_exit, failsafe


def p2f(s):
    try:
        stripped_s = s.strip("%")
        if stripped_s == "":
            return
        else:
            return float(stripped_s) / 100
    except AttributeError:
        return s


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


def data_filter_by_dates(ws, start_date, end_date):
    ws = ws.loc[ws["entry_date"] >= start_date]
    ws = ws.loc[ws["entry_date"] <= end_date]
    return ws


def prepare_data(ws):
    # Convert types
    num_cols = [
        "entry_price_planned",
        "entry_price_actual",
        "exit_price_planned",
        "main_exit_price",
        "threshold_1_expected_price",
        "threshold_1_actual_price",
        "threshold_2_expected_price",
        "threshold_2_actual_price",
        "threshold_3_expected_price",
        "threshold_3_actual_price",
    ]
    ws[num_cols] = ws[num_cols].apply(pd.to_numeric, errors="coerce")

    ws["max_level_reached"] = ws["max_level_reached"].apply(p2f)
    ws["entry_date"] = pd.to_datetime(
        ws["entry_date"], format="%d/%m/%Y", errors="coerce"
    )
    ws["control_exit_date"] = pd.to_datetime(
        ws["control_exit_date"], format="%d/%m/%Y", errors="coerce"
    )

    # Not needed in the new format
    for column in [
        "control_result_%",
        "exit_price_portion",
        "threshold_1_exit_portion",
        "threshold_2_exit_portion",
        "threshold_3_exit_portion",
        "max_level_reached",
    ]:
        ws[column] = ws[column].apply(p2f)

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
        self.all_trades = []  # to derive further metrics
        self.worst_trade_adjusted, self.best_trade_adjusted = 0, 0
        self.balances = dict()
        self.capital_values.append(self.current_capital)
        (
            self.growth,
            self.win_rate,
            self.max_drawdown,
            self.mom_growth,
            self.max_negative_strike,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )
        # We need to have capital part 'snapshot' as of the time of position entry
        self.capital_per_position = dict()
        # For thresholds
        self.left_of_initial_entries = dict()  # list of initial entries
        self.thresholds_reached = dict()  # for the thresholds reached
        self.entry_prices = dict()  # for the entry prices
        self.entry_dates = dict()  # for the entry dates
        # Another dict for capital values and dates detailed
        self.detailed_capital_values = dict()
        # A dict for failed entry days whatever the condition is
        self.failed_entry_day_stocks = dict()
        # For the failsafe checks
        self.failsafe_stock_trigger = dict()
        self.failsafe_active_dates = dict()  # for dates check

    def snapshot_balance(self, current_date_dt):
        self.balances[
            current_date_dt.strftime("%d/%m/%Y")
        ] = self.current_capital  # for the end date
        print("balances:", self.balances)

    def remove_stock_traces(self, stock):
        self.left_of_initial_entries.pop(stock, None)
        self.thresholds_reached.pop(stock, None)
        self.entry_prices.pop(stock, None)
        self.capital_per_position.pop(stock)
        self.thresholds_reached[stock] = []
        self.failed_entry_day_stocks.pop(stock, None)
        self.failsafe_stock_trigger.pop(stock, None)
        self.failsafe_active_dates.pop(stock, None)

def add_entry_no_profit_thresholds(sim, stock):
    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock}")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        sim.capital_per_position[stock] = (
            sim.current_capital / current_simultaneous_positions
        )
        print(
            "-> entry",
            stock,
            "| positions held",
            sim.positions_held,
        )
        print(f"accounting for the brokerage: ${commission}")
        sim.current_capital -= commission
        print(
            f"current_capital: ${sim.current_capital}, allocated to the position: ${sim.capital_per_position[stock]}"
        )


def add_exit_no_profit_thresholds(sim, stock, elem):
    if stock in sim.current_positions:
        sim.current_positions.remove(stock)
        sim.positions_held -= 1

        stock_threshold_results = dict()

        # Calculate result based on three thresholds
        main_part_result = (
            elem["main_exit_price"] - elem["entry_price_actual"]
        ) / elem["entry_price_actual"]

        for i in range(0, 3):
            threshold_result = (
                elem[f"threshold_{i+1}_actual_price"] - elem["entry_price_actual"]
            ) / elem["entry_price_actual"]
            threshold_outcome = threshold_result * elem[f"threshold_{i+1}_exit_portion"]
            threshold_outcome = (
                0 if str(threshold_outcome) == "nan" else threshold_outcome
            )
            stock_threshold_results[i + 1] = threshold_outcome

        result = main_part_result * elem["exit_price_portion"] + sum(
            stock_threshold_results.values()
        )

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

        # Add to all trades
        sim.all_trades.append(result)

        print(
            f"-> exit {stock} | result: {result:.2%} | positions held {sim.positions_held}"
        )
        position_size = sim.capital_per_position[stock]
        print(f"allocated to the position originally: ${position_size}")

        capital_gain = position_size * result
        print(f"capital gain/loss: ${capital_gain}".replace("$-", "-$"))
        print(f"capital state pre exit: ${sim.current_capital}")

        sim.current_capital = sim.current_capital + capital_gain
        print(f"accounting for the brokerage: ${commission}")
        sim.current_capital -= commission
        print(f"balance: ${sim.current_capital}")

        sim.capital_values.append(sim.current_capital)
        sim.capital_per_position.pop(stock)


def median_mom_growth(balances):
    balances = np.array(balances)
    diff_list = np.diff(balances)
    balances_shifted = balances[:-1]
    mom_growth = diff_list / balances_shifted
    return np.median(mom_growth)


def longest_negative_strike(arr):
    # Function to return the longest strike of negative numbers
    max_negative_strike = 0
    for g, k in groupby(arr, key=lambda x: x < 0):
        vals = list(k)
        negative_strike_length = len(vals)
        if g and negative_strike_length > max_negative_strike:
            max_negative_strike = negative_strike_length
    return max_negative_strike


def calculate_metrics(sim, capital):
    print(f"Current capital {sim.current_capital}, starting capital {capital}")
    print(
        f"Positions {current_simultaneous_positions}, tp variant {current_tp_variant_name}"
    )
    sim.growth = (sim.current_capital - capital) / capital
    if sim.winning_trades_number > 0:
        sim.win_rate = (sim.winning_trades_number) / (
            sim.winning_trades_number + sim.losing_trades_number
        )
    else:
        sim.win_rate = 0
    sim.max_drawdown = calculate_max_drawdown(sim.capital_values)
    balances = [v for k, v in sim.balances.items()]
    sim.mom_growth = median_mom_growth(balances)
    sim.max_negative_strike = longest_negative_strike(sim.all_trades)


def print_metrics(sim):
    print()
    print(f"capital growth/loss: {sim.growth:.2%}")
    print(
        f"win rate: {sim.win_rate:.2%} | winning_trades: {sim.winning_trades_number} | losing trades: {sim.losing_trades_number}"
    )
    print(
        f"best trade (adjusted for sizing) {sim.best_trade_adjusted:.2%} | worst trade (adjusted for sizing) {sim.worst_trade_adjusted:.2%}"
    )
    print(f"max_drawdown: {sim.max_drawdown:.2%}")
    print(f"max_negative_strike: {sim.max_negative_strike}")


def update_results_dict(
    results_dict,
    sim,
    current_simultaneous_positions,
    current_variant="control",
    extra_suffix="",
):
    result_current_dict = dict(
        growth=sim.growth * 100,
        win_rate=sim.win_rate * 100,
        winning_trades_number=sim.winning_trades_number,
        losing_trades_number=sim.losing_trades_number,
        best_trade_adjusted=sim.best_trade_adjusted * 100,
        worst_trade_adjusted=sim.worst_trade_adjusted * 100,
        max_drawdown=sim.max_drawdown * 100,
        max_negative_strike=sim.max_negative_strike,
        median_mom_growth=sim.mom_growth * 100,
        simultaneous_positions=current_simultaneous_positions,
        variant_group=current_variant,
    )
    results_dict[
        f"{current_variant}_{current_simultaneous_positions}pos{extra_suffix}"
    ] = result_current_dict
    return results_dict


def add_entry_with_profit_thresholds(sim, stock, entry_price_actual, entry_date_actual):
    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock}")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        sim.capital_per_position[stock] = (
            sim.current_capital / current_simultaneous_positions
        )

        print(
            "-> entry",
            stock,
            "| positions held",
            sim.positions_held,
        )
        print(f"-> entry price: {entry_price_actual}")
        print(f"accounting for the brokerage: ${commission}")
        sim.current_capital -= commission
        print(
            f"current_capital: ${sim.current_capital}, allocated to the position: ${sim.capital_per_position[stock]}"
        )

        # on the entry, we have the full position
        sim.left_of_initial_entries[stock] = 1
        sim.entry_prices[stock] = entry_price_actual
        sim.entry_dates[stock] = entry_date_actual

        # also, on the entry we initiate the dict of thresholds hit for the item
        # they will then be populated with like (0.25, ...)
        sim.thresholds_reached[
            stock
        ] = set()  # appropriate as each would only be there once


def failsafe_trigger_check(sim, stock_prices, current_date_dt):
    for position in sim.current_positions:
        if position not in sim.failsafe_stock_trigger:
            current_df = stock_prices[position][0]
            curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
            failsafe_current_level = sim.entry_prices[position] * (1 + failsafe_trigger_level)
            if not curr_row.empty:
                if curr_row["high"].iloc[0] >= failsafe_current_level:
                    print(f"failsafe level reached for {position} @ {failsafe_current_level}")
                    sim.failsafe_stock_trigger[position] = True
                    sim.failsafe_active_dates[position] = current_date_dt


def failsafe_trigger_rollback(sim, stock_prices, current_date_dt):
    failback_triggers = []
    for position in sim.current_positions:
        current_df = stock_prices[position][0]
        curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
        failsafe_rollback_level = sim.entry_prices[position] * (1 + failsafe_exit_level)
        if not curr_row.empty and (position in sim.failsafe_stock_trigger):
            if sim.failsafe_stock_trigger[position]:
                print(f'{position} failsafe levels check: curr_low {curr_row["low"].iloc[0]} | fsafe level: {failsafe_rollback_level} | failsafe_date {sim.failsafe_active_dates[position]}')
                if (curr_row["low"].iloc[0] < failsafe_rollback_level) and (sim.failsafe_active_dates[position] != current_date_dt):
                    print(f"failsafe rollback for {position} @ {failsafe_rollback_level} on {current_date_dt}")

                    # We should use the correct price as something may just open very low
                    price_to_use = min(curr_row["open"].iloc[0], failsafe_rollback_level)
                    print(f'-- using the price {price_to_use}: as a minimum of {curr_row["open"].iloc[0]} and {failsafe_rollback_level}')

                    failback_triggers.append([position, price_to_use])

    return failback_triggers


def thresholds_check(sim, stock_prices, current_date_dt):
    for position in sim.current_positions:
        current_df = stock_prices[position][0]
        curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
        if not curr_row.empty:
            print(
                f"current high for {position} {curr_row['high'].iloc[0]} | entry: {sim.entry_prices[position]}"
            )
            for each_threshold in current_tp_variant:
                if curr_row["high"].iloc[0] > sim.entry_prices[position] * (
                    1 + each_threshold
                ):
                    # Decrease the residual
                    if each_threshold not in sim.thresholds_reached[position]:
                        sim.left_of_initial_entries[position] -= (
                            1 / current_simultaneous_positions
                        )
                    # Add the threshold
                    sim.thresholds_reached[position].add(each_threshold)
                    print(f"-- {position} reached {each_threshold:.2%}")


def add_exit_with_profit_thresholds(
    sim, stock, entry_price_actual, exit_price_actual, control_result_percent
):
    if stock in sim.current_positions:
        position = stock
        sim.current_positions.remove(stock)
        sim.positions_held -= 1

        # check what thresholds were reached. use entry and exit price and thresholds reached
        number_thresholds_total = len(current_tp_variant)
        number_thresholds_reached = len(sim.thresholds_reached[position])
        divisor = number_thresholds_total + 1
        portion_not_from_thresholds = divisor - number_thresholds_reached

        exit_price_in_calc = exit_price_actual
        print(
            f"exit price used for {position}: {exit_price_in_calc}, entry price: {entry_price_actual}"
        )

        absolute_price_result = (
            exit_price_in_calc - entry_price_actual
        ) / entry_price_actual
        result = absolute_price_result * portion_not_from_thresholds / divisor
        print(
            f"absolute price change result for {position}: {absolute_price_result:.2%} | "
            f"multiplier (considering thresholds): {portion_not_from_thresholds}/{divisor}"
        )
        print(f"relative price change result for {position}: {result:.2%}")
        print(
            f"thresholds reached ({position}): {sim.thresholds_reached[position]}: {number_thresholds_reached} of {number_thresholds_total}"
        )

        # Correct brokerage
        number_brokerage_commissions_paid = number_thresholds_reached + 1

        for threshold_reached in sim.thresholds_reached[position]:
            print(f"--- calc: extra result += {threshold_reached}/{divisor}")
            result += threshold_reached / divisor

        print(f"result ({position}) accounting for thresholds reached: {result:.2%}")

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

        # Add to all trades
        sim.all_trades.append(result)

        print(
            f"-> exit {stock} | result: {result:.2%} | positions held {sim.positions_held}"
        )

        position_size = sim.capital_per_position[stock]
        print(f"allocated to the position originally: ${position_size}")
        capital_gain = position_size * result
        print(f"capital gain/loss: ${capital_gain}".replace("$-", "-$"))
        print(f"capital state pre exit: ${sim.current_capital}")

        sim.current_capital = sim.current_capital + capital_gain
        print(
            f"accounting for the brokerage: ${commission * number_brokerage_commissions_paid} ({commission}x{number_brokerage_commissions_paid})"
        )
        sim.current_capital -= commission * number_brokerage_commissions_paid
        print(f"balance: ${sim.current_capital}")
        sim.capital_values.append(sim.current_capital)

        # Delete from the partial positions left, prices, thresholds for the element
        sim.remove_stock_traces(stock)


# plotting
def plot_latest_sim(latest_sim):
    x, y = [], []
    for key, value in latest_sim.detailed_capital_values.items():
        x.append(key)
        y.append(value)
    _ = plt.plot(x, y)
    ax = plt.gca()
    plt.xticks(fontsize=7)
    lst = list(range(1000))
    lst = lst[0::20]
    for index, label in enumerate(ax.get_xaxis().get_ticklabels()):
        if index not in lst:
            label.set_visible(False)
    plt.show()


def failed_entry_day_check(sim, stock_prices, stock_name, current_date_dt):
    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock_name}")
    else:
        if red_entry_day_exit:
            stock_prices_df = stock_prices[stock_name][0]
            stock_volume_df = stock_prices[stock_name][1]
            curr_state_price = stock_prices_df.loc[
                stock_prices_df["timestamp"] <= current_date_dt
            ]
            curr_state_volume = stock_volume_df.loc[
                stock_volume_df["timestamp"] <= current_date_dt
            ]
            warning, _ = red_day_on_volume(
                curr_state_price, curr_state_volume, output=True, stock_name=stock_name
            )
            if warning:
                sim.failed_entry_day_stocks[stock_name] = True


def failed_entry_day_process(sim, stock_prices, current_date_dt):
    failed_entry_day_stocks_to_iterate = sim.failed_entry_day_stocks.copy()
    for stock_name, elem in failed_entry_day_stocks_to_iterate.items():
        print(f"Failed entry day for {stock_name}, exiting")
        current_df = stock_prices[stock_name][0]
        curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
        if not curr_row.empty:
            print(f"Current open for {stock_name}: {curr_row['open'].iloc[0]}")
            add_exit_with_profit_thresholds(
                sim,
                stock_name,
                sim.entry_prices[stock_name],
                curr_row["open"].iloc[0],
                None,
            )


def get_dates(start_date, end_date):
    try:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("the date must be in the format YYYY-MM-DD")
        exit(0)
    current_date_dt = start_date_dt
    return start_date_dt, end_date_dt, current_date_dt


def interate_over_variant_main_mode(results_dict):
    # Initiate the simulation object
    sim = simulation(capital)

    # Starting for a variant
    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)

    # Balance for the start of the period which will then be updated
    sim.balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
    sim.detailed_capital_values[
        start_date_dt.strftime("%d/%m/%Y")
    ] = sim.current_capital

    # Iterating over days
    while current_date_dt < end_date_dt:

        previous_date_month = current_date_dt.strftime("%m")
        current_date_dt = current_date_dt + timedelta(days=1)
        current_date_month = current_date_dt.strftime("%m")

        if previous_date_month != current_date_month:
            sim.balances[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
        sim.detailed_capital_values[
            current_date_dt.strftime("%d/%m/%Y")
        ] = sim.current_capital

        print(current_date_dt, "| positions: ", sim.current_positions)

        # Entries
        day_entries = ws.loc[ws["entry_date"] == current_date_dt]
        for key, elem in day_entries.iterrows():
            add_entry_no_profit_thresholds(sim, elem["stock"])

        # Exits
        day_exits = ws.loc[ws[f"control_exit_date"] == current_date_dt]
        for key, elem in day_exits.iterrows():
            add_exit_no_profit_thresholds(sim, elem["stock"], elem)

    # Add the final balance at the end of the date
    # sim.snapshot_balance(current_date_dt) # nope, makes mom calc convoluted

    # Calculate metrics and print the results
    calculate_metrics(sim, capital)
    print_metrics(sim)

    # Saving the result in the overall dictionary
    results_dict = update_results_dict(
        results_dict, sim, current_simultaneous_positions
    )
    return results_dict, sim


def iterate_over_variant_tp_mode(results_dict):
    # Initiate the simulation object
    sim = simulation(capital)

    # Starting for a variant
    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)

    # Balance for the start of the period which will then be updated
    sim.balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
    sim.detailed_capital_values[
        start_date_dt.strftime("%d/%m/%Y")
    ] = sim.current_capital

    # Iterating over days
    while current_date_dt < end_date_dt:

        previous_date_month = current_date_dt.strftime("%m")
        current_date_dt = current_date_dt + timedelta(days=1)
        current_date_month = current_date_dt.strftime("%m")

        if previous_date_month != current_date_month:
            sim.balances[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
        sim.detailed_capital_values[
            current_date_dt.strftime("%d/%m/%Y")
        ] = sim.current_capital

        print(
            current_date_dt,
            "| positions: ",
            sim.current_positions,
            "| left of entries:",
            sim.left_of_initial_entries,
        )

        # Prior to looking at entries, process failed entry day stocks
        failed_entry_day_process(sim, stock_prices, current_date_dt)

        # Entries
        day_entries = ws.loc[ws["entry_date"] == current_date_dt]
        for key, elem in day_entries.iterrows():
            add_entry_with_profit_thresholds(
                sim, elem["stock"], elem["entry_price_actual"], elem["entry_date"]
            )

            # On the entry day, check whether the stock is ok or not as this could be used further
            # Note: if the day is red, we should flag that it should be exited on the next
            # In the cycle above, exits from the red closing days should be processed before the entries
            failed_entry_day_check(sim, stock_prices, elem["stock"], current_date_dt)

        # For stocks in positions, check the failsafe protocol
        if failsafe:
            # For each day, check quick n dirty failsafe trigger
            # Checking same day as the entry
            failsafe_trigger_check(sim, stock_prices, current_date_dt)
            # For each day, check failsafe rollback
            # Check next day after the failsafe activation
            failsafe_results = failsafe_trigger_rollback(sim, stock_prices, current_date_dt)
            if failsafe_results != []:
                for elem in failsafe_results:
                    add_exit_with_profit_thresholds(
                        sim,
                        elem[0],
                        sim.entry_prices[elem[0]],
                        elem[1],
                        None,
                    )

        # For each day, need to check the current positions and whether the position reached a threshold
        thresholds_check(sim, stock_prices, current_date_dt)

        # Exits
        day_exits = ws.loc[ws[f"{current_variant}_exit_date"] == current_date_dt]
        for (
            key,
            elem,
        ) in day_exits.iterrows():
            add_exit_with_profit_thresholds(
                sim,
                elem["stock"],
                elem["entry_price_actual"],
                elem["main_exit_price"],
                elem[f"control_result_%"],
            )

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
    return results_dict, sim


def show_monthly_breakdown(result, positions):
    # Open a file for writing
    csv_filename = f'sim_monthly.csv'

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header
        csv_writer.writerow(['Date', 'Value'])

        # Convert string dates to datetime objects
        date_values = {datetime.strptime(date, '%d/%m/%Y'): value for date, value in result.items()}

        print()
        print('(!) Note: monthly results are only shown for the last simulation')
        print('--- Monthly breakdown ---')

        # Iterate over the dictionary and print values at the beginning of each month
        current_month = None
        for date, value in sorted(date_values.items()):
            if date.month != current_month:
                current_month = date.month
                formatted_date = date.replace(day=1).strftime("%d/%m/%Y")
                rounded_value = round(value, 1)
                print(f'{formatted_date}: {rounded_value}')
                csv_writer.writerow([formatted_date, rounded_value])

        print('-------------------------')
        print(f'(i) results have been written to {csv_filename}')
        print()

if __name__ == "__main__":

    arguments = define_args()
    red_entry_day_exit, failsafe = process_filter_args()

    print("reading the values...")

    # Dates
    start_date = arguments["start"]
    end_date = arguments["end"]
    reporting_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(
        days=2 * 365
    )
    # ^^^ -2 years ago from start is ok
    reporting_start_date = reporting_start_date.strftime("%Y-%m-%d")

    # This is working ok
    exchange = arguments["exchange"]
    ws = gsheetsobj.sheet_to_df(config["logging"]["gsheet_name"], f"{exchange}")
    ws.columns = config["logging"]["gsheet_columns"]
    ws = prepare_data(ws)

    # Dict to hold all the results
    results_dict = dict()

    # Dates filtering for the dataset
    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)
    ws = data_filter_by_dates(ws, start_date_dt, end_date_dt)

    # > Iterating through days and variants for the fixed TP levels per the control & spreadsheet
    current_tp_variant_name = None
    if arguments["mode"] == "main":
        for current_simultaneous_positions in simultaneous_positions:
            results_dict, latest_sim = interate_over_variant_main_mode(
                results_dict
            )

            # For the main mode, get data on MoM performance
            if arguments["show_monthly"]:
                show_monthly_breakdown(latest_sim.detailed_capital_values, positions=simultaneous_positions)

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
            stock_info = get_stock_data(f"{stock}{suffix}", reporting_start_date)
            stock_prices[stock] = stock_info

        for current_tp_variant_name, current_tp_variant in take_profit_variants.items():
            print(
                f">> starting the variant {current_tp_variant_name}-{current_tp_variant}"
            )

            for current_simultaneous_positions in simultaneous_positions:
                results_dict, latest_sim = iterate_over_variant_tp_mode(results_dict)

            print(
                f">> finished the variant {current_tp_variant_name}-{current_tp_variant}"
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
            "max_negative_strike",
            "win_rate",
            "median_mom_growth",
            "winning_trades_number",
            "worst_trade_adjusted",
        ]
    ]

    # just an info bit
    if arguments["mode"] == "tp":
        print(f"take profit variants tested: {take_profit_variants}")

    # save to csv
    final_result.to_csv("sim_summary.csv", index=False)
    print()
    print("(i) summary saved to sim_summary.csv")

    # plotting
    if arguments["plot"]:
        plot_latest_sim(latest_sim)
