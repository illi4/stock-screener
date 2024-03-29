### Note: works with the R&D spreadsheet format (Trading journal R&D 2021)
# Simulates trading progress and results over time using a spreadsheet with various considerations
# Will save the result to simulator_result.csv
# Use example: python simulator.py -mode=main -exchange=asx -start=2021-02-01 -end=2021-07-01

import libs.gsheetobj as gsheetsobj
from libs.signal import red_day_on_volume
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from libs.stocktools import get_stock_suffix, get_stock_data
import argparse
from itertools import groupby
from libs.signal import market_bearish

parser = argparse.ArgumentParser()
import matplotlib.pyplot as plt

# Settings (default)
confidence_filter = [8, 9]
penny_filter = ["Y", "N"]
capital = 5000
commission = 1  # this is brokerage (per entry / per exit); IBKR is very cheap

'''
higher_or_equal_open_filter, higher_strictly_open_filter, and red_entry_day_exit are defined in a function
'''

# Quick and dirty check of the 'failsafe level' hypothesis
failsafe_trigger_level = 0.15
failsafe_exit_level = 0.05

# Ad hoc settings for testing a few hypothesis:
# Decreasing volume, sustainable price movement, pre formation run up value
condition_decreasing_volume_formation = False
condition_sustainable_price_growth = False
pre_formation_level_lowest_value = 0
pre_formation_level_highest_value = 10000

# Ad Hoc 2
check_stochastic = True
check_stochastic_threshold = 0.9


gsheet_name = 'Trading journal R&D 2021'  # hardcoded legacy name

# Variations to go through
simultaneous_positions = [4] #[3, 4, 5]
variant_names = ["control", "test_a", "test_b", "test_c", "test_e"]
tp_base_variant = "control"  # NOTE: works with control and test_c currently (need to have the price column)

# Red entry day on volume check
# red_entry_day_exit = True # is defined in a function

# Take profit level variations
# Would be used iterating over control with simultaneous_positions variations too
take_profit_variants = {
    #"_repeated_to_control": [0.25, 0.45, 0.9],
    #"tp_b": [0.5, 1],
    #"tp_c": [0.15, 0.5, 0.9, 1.75],
    "tp_d": [0.5, 1, 1.5],
    #"tp_d_a": [0.2, 1, 1.5],
    #"tp_d_b": [0.25, 1, 1.5],
    #"tp_e": [0.25, 0.45, 0.9, 1.45],
    #"tp_g": [0.25, 0.9, 1.45, 1.75],
    #"tp_h": [1.45, 1.75, 1.95],
    #"tp_k1": [0.45, 1.75, 1.95],
    #"tp_l1": [0.45, 1.45, 1.75, 1.95],
    #"tp_x": [0.1, 1.45, 1.75],
    #"tp_y": [0.1, 1.75, 1.95]
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
    "higher_strictly_open",
    "control_exit_date",
    "exit_price_planned",
    "control_price",
    "outcome",
    "control_result_%",
    "test_a_exit_date",
    "test_a_result_%",
    "test_b_exit_date",
    "test_b_result_%",
    "test_c_exit_date",
    "test_c_price",
    "test_c_result_%",
    "test_e_exit_date",
    "test_e_result_%",
    "max_level_reached",
    "comments",
    "time_in_trade_control",
    "time_in_trade_test_a",
    "time_in_trade_test_b",
    "time_in_trade_test_c",
    "time_in_trade_test_e",
    "pre_formation_run_up",
    "sustainable_growth",
    "decreasing_volume_in_formation"
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
        "--nofilter", action="store_true", help="No filters on the launch"
    )
    parser.add_argument("--he", action="store_true", help="Higher or equal opens only")
    parser.add_argument("--ho", action="store_true", help="Higher opens only")
    parser.add_argument("--market", action="store_true", help="Consider market")
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

    return arguments


def process_filter_args():

    # Default values if no rule
    higher_or_equal_open_filter = ["Y", "N"]
    higher_strictly_open_filter = ["Y", "N"]

    if arguments["nofilter"]:
        higher_or_equal_open_filter = ["Y", "N"]
        higher_strictly_open_filter = ["Y", "N"]
    if arguments["he"]:
        higher_or_equal_open_filter = ["Y"]
        higher_strictly_open_filter = ["Y", "N"]
    if arguments["ho"]:
        higher_or_equal_open_filter = ["Y"]
        higher_strictly_open_filter = ["Y"]
    if arguments["red_day_exit"]:
        red_entry_day_exit = True
    else:
        red_entry_day_exit = False
    if arguments["market"]:
        market_consideration = True
    else:
        market_consideration = False

    failsafe = True if arguments["failsafe"] else False

    return higher_or_equal_open_filter, higher_strictly_open_filter, red_entry_day_exit, market_consideration, failsafe


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


def data_filter_by_dates(ws, start_date, end_date):
    ws = ws.loc[ws["entry_date"] >= start_date]
    ws = ws.loc[ws["entry_date"] <= end_date]
    return ws


def prepare_data(ws):
    # Convert types
    num_cols = [
        "confidence",
        "entry_price_planned",
        "entry_price_actual",
        "exit_price_planned",
        "control_price",
        "test_c_price",
        "pre_formation_run_up"
    ]
    ws[num_cols] = ws[num_cols].apply(pd.to_numeric, errors="coerce")

    # Apply filters
    ws = ws.loc[ws["confidence"].isin(confidence_filter)]
    ws = ws.loc[ws["penny_stock"].isin(penny_filter)]
    ws = ws.loc[ws["higher_open"].isin(higher_or_equal_open_filter)]
    ws = ws.loc[ws["higher_strictly_open"].isin(higher_strictly_open_filter)]

    # Ad hoc checks
    if condition_decreasing_volume_formation:
        ws = ws.loc[ws["decreasing_volume_in_formation"].isin(['Y'])]
    if condition_sustainable_price_growth:
        ws = ws.loc[ws["sustainable_growth"].isin(['Y'])]
    ws = ws.loc[ws["pre_formation_run_up"] >= pre_formation_level_lowest_value]
    ws = ws.loc[ws["pre_formation_run_up"] <= pre_formation_level_highest_value]

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
        # quick one
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
            entry_date_actual, " "
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
        current_df = stock_prices[position][0].copy()
        current_df['next_open'] = current_df['open'].shift(-1)

        curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]

        failsafe_rollback_level = sim.entry_prices[position] * (1 + failsafe_exit_level)
        if not curr_row.empty and (position in sim.failsafe_stock_trigger):
            if sim.failsafe_stock_trigger[position]:
                print(f'{position} failsafe levels check: curr_low {curr_row["low"].iloc[0]} | fsafe level: {failsafe_rollback_level} | failsafe_date {sim.failsafe_active_dates[position]}')
                if (curr_row["low"].iloc[0] < failsafe_rollback_level) and (sim.failsafe_active_dates[position] != current_date_dt):
                    print(f"failsafe rollback for {position} @ {failsafe_rollback_level} on {current_date_dt}")

                    # We should use the correct price
                    #price_to_use = min(curr_row["open"].iloc[0], failsafe_rollback_level) # old incorrect logic
                    price_to_use = curr_row["next_open"].iloc[0]
                    print(f'-- using the price {price_to_use}: next day open')

                    failback_triggers.append([position, price_to_use])

    return failback_triggers


def thresholds_check(sim, stock_prices, current_date_dt):
    for position in sim.current_positions:
        current_df = stock_prices[position][0]

        if current_df.empty:
            print('Error - no price data')
            exit(0)

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
        print(
            f"relative price change result for {position}: {result:.2%}"
        )
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
            f"accounting for the brokerage: ${commission*number_brokerage_commissions_paid} ({commission}x{number_brokerage_commissions_paid})"
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



# calculating RSI (gives the same values as TradingView)
# https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
def RSI(series, period=14):
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean()
    return 100 - 100 / (1 + rs)


# calculating Stoch RSI (gives the same values as TradingView)
# https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
def StochRSI(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean()
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi_K.iloc[-1], stochrsi_D.iloc[-1]

def stochastic_is_ok(sim, stock_prices, stock_name, current_date_dt):
    print(stock_name)
    stock_prices_df = stock_prices[stock_name][0]

    current_df = stock_prices_df.loc[
        stock_prices_df["timestamp"] <= current_date_dt
    ].copy()

    oscillator_values = StochRSI(current_df['close'])
    oscillator_max = max(oscillator_values)
    print('Oscillator value', oscillator_max)

    if oscillator_max >= check_stochastic_threshold:
        print('Oscillator too high')
        return False
    else:
        return True


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
        day_exits = ws.loc[ws[f"{current_variant}_exit_date"] == current_date_dt]
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
    return results_dict, sim


def get_market_state(current_date_dt):
    market_price, market_volume = market_info[0], market_info[1]
    # has to be strict condition as we only know data for the previous day
    curr_state_market_price = market_price.loc[
        market_price["timestamp"] < current_date_dt
        ]
    curr_state_market_volume = market_volume.loc[
        market_volume["timestamp"] < current_date_dt
        ]
    is_market_bearish, _ = market_bearish(curr_state_market_price, curr_state_market_volume, output=True)
    return is_market_bearish


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

            # Firstly, check whether the market performance is Ok
            if market_consideration:
                is_market_bearish = get_market_state(current_date_dt)

            # Check stoch RSI
            if check_stochastic:
                is_stochastic_ok = stochastic_is_ok(sim, stock_prices, elem["stock"], current_date_dt)
            else:
                is_stochastic_ok = True

            # Add the entry
            if ((market_consideration and not is_market_bearish) or (not market_consideration)) and is_stochastic_ok:
                add_entry_with_profit_thresholds(
                    sim, elem["stock"], elem["entry_price_actual"], elem["entry_date"]
                )

            # On the entry day, check whether the stock is ok or not as this could be used further
            # Note: if the day is red, we should flag that it should be exited on the next
            # In the cycle above, exits from the red closing days should be processed before the entries
            failed_entry_day_check(sim, stock_prices, elem["stock"], current_date_dt)

        # For stocks in positions
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
                elem[f"entry_price_actual"],
                elem[f"{current_variant}_price"],
                elem[f"{current_variant}_result_%"],
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


if __name__ == "__main__":

    arguments = define_args()
    (
        higher_or_equal_open_filter,
        higher_strictly_open_filter,
        red_entry_day_exit,
        market_consideration,
        failsafe
    ) = process_filter_args()

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
    ws = gsheetsobj.sheet_to_df(gsheet_name, f"{exchange}")
    ws.columns = sheet_columns
    ws = prepare_data(ws)


    # Dict to hold all the results
    results_dict = dict()

    # Dates filtering for the dataset
    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)
    ws = data_filter_by_dates(ws, start_date_dt, end_date_dt)

    # > Iterating through days and variants for the fixed TP levels per the control & spreadsheet
    current_tp_variant_name = None
    if arguments["mode"] == "main":
        for current_variant in variant_names:

            print(f">> starting the variant {current_variant}")

            for current_simultaneous_positions in simultaneous_positions:
                results_dict, latest_sim = interate_over_variant_main_mode(results_dict)

            print(f">> finished the variant {current_variant}")

    # < Finished iterating

    # > Iterating through days and take profit variants for the dynamic TP levels
    # Only supported for control but allows to make some conclusions too
    if arguments["mode"] == "tp":
        current_variant = tp_base_variant
        stock_names = [item.stock for key, item in ws.iterrows()]
        stock_prices = dict()
        suffix = get_stock_suffix(exchange)

        # Getting market data
        market_info = get_stock_data(f"AXJO.INDX", reporting_start_date)

        # Getting individual stocks data
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
    final_result.to_csv("simulator_result.csv", index=False)
    print("results saved to simulator_result.csv")

    # plotting
    if arguments["plot"]:
        plot_latest_sim(latest_sim)
