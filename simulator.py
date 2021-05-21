# Simulates trading progress and results over time using a spreadsheet with various considerations
import libs.gsheetobj as gsheetsobj
from libs.settings import gsheet_name
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from libs.stocktools import get_stock_suffix, get_stock_data

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
end_date = "2021-05-20"

# Take profit level variations
# Would be used iterating over control with simultaneous_positions variations too
# Control is 0.25, 0.5, 0.9
take_profit_variants = {
    'tp_a':[0.25, 0.5, 0.9],  # this is just to check that I get the same result as in control, to remove
    'tp_x':[0.15, 0.5, 0.9],  # this - to remove
    #'tp_a':[0.3, 0.5, 0.9],
    #'tp_b':[0.5, 1],
    #'tp_c':[0.15, 0.5, 0.9],
    #'tp_d':[0.5, 1, 1.5],
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
    #for current_variant in variant_names:
    #for current_simultaneous_positions in simultaneous_positions:

    ###TO TEST. ADD THE CYCLE BACK.
    current_variant = 'control'
    current_simultaneous_positions = 3

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

    #### Iteration finished

    ##### More complex implementation - various options on threshold levels # TODO FINISH - FOLLOWING STEPS HERE
    # Need to grab prices for all stocks involved in the period
    stock_names = [item.stock for key, item in ws.iterrows()]
    stock_prices = dict()
    suffix = get_stock_suffix(exchange)
    for stock in stock_names:
        print(f'getting data for {stock}{suffix}')
        stock_info = get_stock_data(f'{stock}{suffix}')
        stock_prices[stock] = stock_info


    # >>> Similar iterations like for control, but with dynamic thresholds #HERE#
    # DOESNT SEEM TO BE WORKING FINE, CONTINUE CHECKING WHY
    # Need to track the entry price and exit with a part of a position
    # ...
    current_tp_variant = [0.25, 0.45, 0.9]  # this could have 3 or more values, would need to maintain dict per asset
    current_tp_variant_name = 'xxx'
    current_simultaneous_positions = 3  # for testing

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
    print(f"simulating control with TP levels {current_tp_variant}")
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    current_date_dt = start_date_dt

    # Balance for the start of the period which will then be updated
    balances = dict()
    balances[start_date_dt.strftime("%d/%m/%Y")] = current_capital

    # How much of a position is left and entry prices - this is required for partial exits tracking
    left_of_initial_entries = dict()
    thresholds_reached = dict()  # for the thresholds reached
    entry_prices = dict() # here

    # Iterating over days
    while current_date_dt < end_date_dt:

        previous_date_month = current_date_dt.strftime("%m")
        current_date_dt = current_date_dt + timedelta(days=1)
        current_date_month = current_date_dt.strftime("%m")

        if previous_date_month != current_date_month:
            balances[current_date_dt.strftime("%d/%m/%Y")] = current_capital

        print(current_date_dt, "| positions: ", current_positions, "| left of entries:", left_of_initial_entries)

        # Entries
        day_entries = ws.loc[ws["entry_date"] == current_date_dt]
        for key, elem in day_entries.iterrows():

            if len(current_positions) + 1 > current_simultaneous_positions:
                print(f"max possible positions held, skipping {elem['stock']}")
            else:
                positions_held += 1
                current_positions.add(elem["stock"])
                print("-> entry", elem["stock"], "| positions held", positions_held)
                print(f"-> entry price: {elem['entry_price_actual']}")
                print(f"accounting for the trade price: ${commission}")
                current_capital -= commission
                # on the entry, we have the full position
                left_of_initial_entries[elem["stock"]] = 1
                entry_prices[elem["stock"]] = elem['entry_price_actual']
                # also, on the entry we initiate the dict of thresholds hit for the item
                # they will then be populated with like (0.25, ...)
                thresholds_reached[elem["stock"]] = set()  # appropriate as each would only be there once


        # For each day, need to check the current positions and whether the position reached a threshold
        # Note: also need to consider the entry date
        for position in current_positions:
            current_df = stock_prices[position][0]
            curr_row = current_df.loc[current_df["timestamp"] == current_date_dt]
            if not curr_row.empty:
                print(f"Current high for {position} {curr_row['high'].iloc[0]} | entry: {entry_prices[position]}") # to continue here
                for each_threshold in current_tp_variant:
                    #print(f"Checking the {each_threshold} threshold: {curr_row['high'].iloc[0]} vs {entry_prices[position]*(1+each_threshold)}")
                    if curr_row['high'].iloc[0] > entry_prices[position]*(1+each_threshold):
                        thresholds_reached[position].add(each_threshold)
                        print("-- reached", each_threshold)

        # Exits
        day_exits = ws.loc[ws[f"control_exit_date"] == current_date_dt]
        for key, elem in day_exits.iterrows():  ### this needs to be modified to account for partial exits
            if elem["stock"] in current_positions:
                position = elem["stock"]
                current_positions.remove(elem["stock"])
                positions_held -= 1

                # check what thresholds were reached. use entry and exit price and thresholds reached
                print(thresholds_reached[position])
                number_thresholds_total = len(current_tp_variant)
                number_thresholds_reached = len(thresholds_reached[position])
                divisor = number_thresholds_total+1
                portion_not_from_thresholds = divisor - number_thresholds_reached

                absolute_price_result = (elem[f"exit_price_actual"] - elem[f"entry_price_actual"]) / elem[
                    f"entry_price_actual"]
                result = absolute_price_result * portion_not_from_thresholds / divisor
                print(f'Price change result for {position}: {result}')
                print(
                    f"Thresholds reached ({position}): {thresholds_reached[position]}: {number_thresholds_reached} of {number_thresholds_total}")

                for threshold_reached in thresholds_reached[position]:
                    result += threshold_reached/divisor

                print(f"Result ({position}) accounting for thresholds reached: {result}")

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
                        print(f"best_trade_adjusted is now {best_trade_adjusted}")
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

                # old logic
                current_capital = current_capital * (
                        1
                        + elem[f"control_result_%"]
                        / current_simultaneous_positions
                )
                print(f"accounting for the trade price: ${commission}")
                current_capital -= commission
                print(f"balance: ${current_capital}")
                capital_values.append(current_capital)

                # delete from the partial positions left, prices, thresholds for the element
                left_of_initial_entries.pop(elem["stock"], None)
                thresholds_reached.pop(elem["stock"], None)
                entry_prices.pop(elem["stock"], None)
                thresholds_reached[elem["stock"]]=[]

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
        variant_group='control'
    )
    results_dict[f"control_{current_simultaneous_positions}pos_{current_tp_variant_name}"] = result_current_dict

    ## <<< Finished more complex iterations - finalise iterating

    ######### Finalisation
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