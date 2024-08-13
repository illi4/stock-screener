#########################################
#                                       #
#         IMPORTS AND SETUP 🐍          #
#                                       #
#########################################

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

from statistics import mean
from peewee import IntegrityError
import statistics
from collections.abc import Mapping

import libs.gsheetobj as gsheetsobj
from datetime import datetime, timedelta
from libs.stocktools import get_stock_data, Market
from libs.techanalysis import fisher_distance
from libs.stocktools import get_stock_data
from libs.db import get_historical_prices

import argparse
import pandas as pd
import random


parser = argparse.ArgumentParser()

from libs.read_settings import read_config
config = read_config()

#from libs.signal import red_day_on_volume # not used
from libs.simulation import Simulation
from libs.db import check_earliest_price_date, delete_all_prices, bulk_add_prices, get_price_from_db
from libs.helpers import create_report, define_simulator_args, data_filter_by_dates, prepare_data, filter_dataframe, data_filter_from_date

pd.set_option("display.max_columns", None)

#########################################
#                                       #
#         TRADE PROCESSING 💼           #
#                                       #
#########################################

def get_lowest_price_before_entry(stock, entry_date):
    previous_date = entry_date - timedelta(days=1)
    price_info = get_price_from_db(stock, previous_date)
    return price_info['low']

def get_highest_body_level(stock, entry_date):
    previous_date = entry_date - timedelta(days=1)  # this is the bullish reference candle
    price_info = get_price_from_db(stock, previous_date)
    return max(price_info['close'], price_info['open'])

def get_next_opening_price(stock, current_date):
    next_date = current_date + timedelta(days=1)  # looking at the next day open
    price_info = get_price_from_db(stock, next_date, look_backwards=False)
    return price_info['open']

def average_dict_values(dict_list):
    if not dict_list:
        return {}

    keys = dict_list[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in dict_list if key in d]
        if values:
            if isinstance(values[0], Mapping):
                result[key] = average_dict_values(values)
            elif isinstance(values[0], (int, float)):
                result[key] = statistics.mean(values)
            else:
                result[key] = values[0]  # For non-numeric values, just take the first one
    return result


def randomly_exclude_rows(df, exclusion_rate):
    """
    Randomly exclude a percentage of rows from the dataframe.

    :param df: Input dataframe
    :param exclusion_rate: Percentage of rows to exclude (as decimal)
    :return: Dataframe with randomly excluded rows
    """
    num_rows = len(df)
    num_to_exclude = int(num_rows * exclusion_rate)
    exclude_indices = random.sample(range(num_rows), num_to_exclude)
    return df.drop(df.index[exclude_indices])


def get_reference_dates(start_date, df):
    start_date = pd.to_datetime(start_date)
    future_dates = df[df['entry_date'] > start_date]['entry_date'].sort_values().unique()

    # Select the first 10 unique dates (including start_date)
    candidate_dates = [start_date] + list(future_dates[:9])

    # Randomly sample the number of dates specified in the config
    sample_size = min(config["simulator"]["sample_size"], len(candidate_dates))
    reference_dates = random.sample(candidate_dates, sample_size)

    return sorted(reference_dates)  # Return sorted dates for chronological order


def average_results(results_list):
    averaged_results = {}
    for key in results_list[0].keys():
        values = [result[key] for result in results_list]
        if isinstance(values[0], (int, float)):
            averaged_results[key] = mean(values)
        else:
            averaged_results[key] = values[0]  # For non-numeric values, just take the first one
    return averaged_results

def process_entry(sim, stock, entry_price, take_profit_variant, current_date, initial_stop, close_higher_percentage, stop_below_bullish_reference_variant):

    if len(sim.current_positions) + 1 > sim.current_simultaneous_positions:
        print(f"(i) max possible positions | skipping {stock} entry")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        sim.capital_per_position[stock] = sim.current_capital / sim.current_simultaneous_positions

        # Set fisher-related info for a stock
        sim.set_fisher_distance_profit_info(stock)

        # Set take profit levels
        sim.set_take_profit_levels(stock, take_profit_variant, entry_price)

        # Set initial entry and stop / allocation reference prices 
        allocation_reference_price = get_highest_body_level(stock, current_date)
        lowest_price_before_entry = get_lowest_price_before_entry(stock, current_date)
        adjusted_stop_reference = lowest_price_before_entry * (1-stop_below_bullish_reference_variant)

        sim.set_initial_entry(stock, entry_price,
                              config["simulator"]["first_entry_allocation"],
                              close_higher_percentage,
                              allocation_reference_price, 
                              adjusted_stop_reference
                              )

        # Show info
        print(f"-> ENTER {stock} | positions held: {sim.positions_held}")
        print(f'-- commission ${config["simulator"]["commission"]}')
        print(f"-- current capital on entry: ${sim.current_capital}, allocated to the position: ${sim.capital_per_position[stock]}")
        # Create a string of take profit prices
        tp_prices = " | ".join([f"${level['price']:.2f}" for level in sim.take_profit_info[stock]['levels']])
        print(f"-- lowest price before entry (on the reference candle): ${lowest_price_before_entry:.2f}")
        print(f"-- adjusted stop reference (to move to after 2nd entry): ${adjusted_stop_reference:.2f}")
        print(f"-- take profit levels: {tp_prices}")


        # Set the stop level using the value from the sheet
        sim.set_stop_loss(stock, initial_stop)
        sim.current_capital -= config["simulator"]["commission"]


def exit_all_positions(sim, current_date_dt):
    for stock in list(sim.current_positions):  # Use list() to avoid modifying set during iteration
        price_data = get_price_from_db(stock, current_date_dt)
        if price_data:
            process_exit(sim, stock, price_data)
        else:
            print(f"Warning: No price data found for {stock} on {current_date_dt}. Skipping exit.")

def calculate_profit_contribution(take_profit_info):
    return sum(
        float(level['actual_level']) * float(level['exit_proportion'].strip('%')) / 100  # proportion is in string from config
        for level in take_profit_info['levels']
        if level['reached']
    )

def calculate_fisher_contribution(sim, stock):
    entry_price = sim.get_average_entry_price(stock)
    data = sim.fisher_distance_exits[stock]
    contribution = 0

    for i, exit_price in enumerate(data['prices']):
        move_percentage = (exit_price - entry_price) / entry_price
        exit_proportion = data['used_proportion'] / data['number_exits']  # Assuming equal proportion for each exit
        contribution += move_percentage * exit_proportion

    return contribution

def process_exit(sim, stock_code, price_data, forced_price=None):
    if stock_code in sim.current_positions:

        entry_price = sim.get_average_entry_price(stock_code)  # taking an average of entries
        total_position_size = sim.capital_per_position[stock_code]
        position_allocation = sim.entry_allocation[stock_code]  # it could be possible that we only entered for 50%

        # Check if we need to use specific price
        exit_price = forced_price if forced_price is not None else price_data['open']

        final_exit_proportion = 1 - sim.take_profit_info[stock_code]['taken_profit_proportion'] - sim.fisher_distance_exits[stock_code]['used_proportion']
        final_price_change = (exit_price - entry_price) / entry_price

        # Print some stats
        print(f"-> Exit ({stock_code}) [${total_position_size:.2f} position size] | allocation {position_allocation:.0%}")
        print(f"-- exit price: ${exit_price:.2f} | entry ${entry_price:.2f} | change {final_price_change:.2%}")
        print(f"-- proportion used by taking profit at fixed levels: {sim.take_profit_info[stock_code]['taken_profit_proportion']:.0%}")
        print(f"-- proportion used by taking profit via Fisher Distance: {sim.fisher_distance_exits[stock_code]['used_proportion']:.0%}")

        # Calculate the contribution from the Fisher distance crossing
        fisher_contribution = calculate_fisher_contribution(sim, stock_code)
        print(f'-- position PNL from Fisher Distance crossing TP: {fisher_contribution:.2%}')

        # Calculate the contribution from the take profit levels
        # This is representing the growth from the overall original amount, accounted for the proportion of TP levels
        tp_contribution = calculate_profit_contribution(sim.take_profit_info[stock_code])
        print(f'-- position PNL from take profit levels: {tp_contribution:.2%}')

        # Print reached levels
        reached_levels = [level for level in sim.take_profit_info[stock_code]['levels'] if level['reached']]
        for level in reached_levels:
            print(f"  ├ level of {level['level']} [v]")

        # Calculate the final exit part contribution
        last_exit_contribution = final_price_change*final_exit_proportion
        print(f'-- position PNL from the exit: {last_exit_contribution:.2%}')

        overall_result = tp_contribution + last_exit_contribution + fisher_contribution
        print(f'-- total position PNL before accounting for allocation: {overall_result:.2%}')

        final_outcome = overall_result*position_allocation
        print(f'-- total position PNL (accounted for allocation): {final_outcome:.2%}')

        # Calculate the outcome using the original position size
        profit_amount = total_position_size * (final_outcome)
        print(f'--> profit/loss ${profit_amount:.2f}')

        previous_capital = sim.current_capital
        sim.current_capital += profit_amount
        sim.current_capital -= config["simulator"]["commission"]
        print(f"Capital ${previous_capital:.2f} -> ${sim.current_capital:.2f}")

        # Delete traces
        sim.current_positions.remove(stock_code)
        sim.positions_held -= 1
        sim.remove_stock_traces(stock_code)

        # Results update
        sim.update_capital(sim.current_capital)
        sim.update_trade_statistics(overall_result, sim.current_simultaneous_positions)


def update_results_dict(
    results_dict,
    sim,
    current_simultaneous_positions,
    take_profit_variant_name,
    close_higher_percentage_variant_name,
    stop_below_bullish_reference_variant,
    current_variant="control",
    extra_suffix="",
):
    result_current_dict = dict(
        growth=sim.growth,
        win_rate=sim.win_rate,
        winning_trades_number=sim.winning_trades_number,
        losing_trades_number=sim.losing_trades_number,
        best_trade_adjusted=sim.best_trade_adjusted,
        worst_trade_adjusted=sim.worst_trade_adjusted,
        max_drawdown=sim.max_drawdown,
        max_negative_strike=sim.max_negative_strike,
        median_mom_growth=sim.mom_growth,
        average_mom_growth=sim.average_mom_growth,
        max_positions=current_simultaneous_positions,
        variant_group=current_variant,
    )
    results_dict[
        f"{current_variant}_{current_simultaneous_positions}pos_" \
        f"{take_profit_variant_name}_chp{close_higher_percentage_variant_name}{extra_suffix}_stp{stop_below_bullish_reference_variant}"
    ] = result_current_dict
    return results_dict


def get_dates(start_date, end_date):
    try:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("the date must be in the format YYYY-MM-DD")
        exit(0)
    current_date_dt = start_date_dt
    return start_date_dt, end_date_dt, current_date_dt


def check_profit_levels(sim, current_date_dt, take_profit_variant):
    for stock in sim.current_positions:
        price_data = get_price_from_db(stock, current_date_dt)
        if price_data:
            sim.check_and_update_take_profit(stock, price_data['high'], price_data['open'], take_profit_variant, config["simulator"]["commission"])


def check_fisher_based_take_profit(sim, current_date_dt, date_changed_reported):
    if date_changed_reported:
        print('(i) skipping fisher distance check because of the weekend or holiday')
        return

    reentry_threshold = config["simulator"]["fisher_distance_exit"]["reentry_threshold"]
    minimum_price_increase = config["simulator"]["fisher_distance_exit"]["minimum_price_increase"]

    for stock in sim.current_positions:
        historical_prices = get_historical_prices(stock, current_date_dt)
        last_price_date = historical_prices.index[-1]

        if (stock not in sim.last_fisher_calculation or
                sim.last_fisher_calculation[stock] is None or
                'date' not in sim.last_fisher_calculation[stock] or
                last_price_date > sim.last_fisher_calculation[stock]['date']):
            fisher_df = fisher_distance(historical_prices)
            current_fisher_dist = fisher_df['distance'].iloc[-1]
            previous_fisher_dist = fisher_df['distance'].iloc[-2]

            sim.last_fisher_calculation[stock] = {
                'date': last_price_date,
                'current': current_fisher_dist,
                'previous': previous_fisher_dist
            }

        else:
            current_fisher_dist = sim.last_fisher_calculation[stock]['current']
            previous_fisher_dist = sim.last_fisher_calculation[stock]['previous']

        print(f"-- fisher distance value ({stock}): {current_fisher_dist:.4f}")

        if (current_fisher_dist > reentry_threshold) and not sim.fisher_distance_above_threshold[stock]:
            sim.fisher_distance_above_threshold[stock] = True
            print(f"-> Fisher distance for {stock} went above reentry threshold: {current_fisher_dist:.4f}")

        if (previous_fisher_dist > config["simulator"]["fisher_distance_exit"]["crossed_down_value"]
                and current_fisher_dist <= config["simulator"]["fisher_distance_exit"]["crossed_down_value"]):
            if sim.fisher_distance_above_threshold[stock] and sim.fisher_distance_exits[stock]['number_exits'] < config["simulator"]["fisher_distance_exit"]["max_exits"]:
                # Check if the price has increased by the minimum required percentage
                entry_price = sim.get_average_entry_price(stock)
                current_price = historical_prices['close'].iloc[-1]
                price_increase = (current_price - entry_price) / entry_price

                if price_increase >= minimum_price_increase:
                    print(f"-> Fisher distance for {stock} crossed below zero (from {previous_fisher_dist:.4f} to {current_fisher_dist:.4f})")
                    next_day_dt = current_date_dt + timedelta(days=1)
                    price_data = get_price_from_db(stock, next_day_dt)
                    print(f"-- will take profit at the next day open price {price_data['open']}")
                    sim.check_and_update_fisher_based_profit(stock, price_data['open'],
                                                             config["simulator"]["fisher_distance_exit"]["exit_proportion"],
                                                             config["simulator"]["commission"])
                    sim.fisher_distance_above_threshold[stock] = False
                else:
                    print(f"Skipping Fisher distance exit for {stock}: price increase ({price_increase:.2%}) is below the minimum required ({minimum_price_increase:.2%})")
            else:
                print(f"Skipping Fisher distance exit for {stock}: {'threshold not reached' if not sim.fisher_distance_above_threshold[stock] else 'max exits reached'}")

def check_stop_loss(sim, current_date_dt):

    stops_hit = []

    for stock in sim.current_positions:
        price_data = get_price_from_db(stock, current_date_dt)
        # check if the low for the day is below stop loss level
        stop_loss_hit = (price_data['low'] < sim.stop_loss_prices[stock])
        if stop_loss_hit:
            # calculate which price to use. some stocks gap down significantly
            stopped_out_price = min(sim.stop_loss_prices[stock], price_data['open'])

            # different messaging if we have stop loss vs trailing stop
            msg = 'TRAILING PROFIT' if sim.trailing_stop_active.get(stock, False) else 'STOP LOSS'
            print(f'-> {msg} HIT ({stock}) @ ${stopped_out_price:.2f}')
            # add this to stops_hit
            stops_hit.append(dict(stock=stock, stopped_out_price=stopped_out_price, price_data=price_data))

    # Process exits for the stocks
    for hits in stops_hit:
        process_exit(sim, hits['stock'], hits['price_data'], forced_price=hits['stopped_out_price'])


def check_stop_breakeven(sim, current_date_dt):
    stops_hit = []

    for stock in sim.current_positions:
        if stock in sim.breakeven_stop_loss_prices:
            price_data = get_price_from_db(stock, current_date_dt)
            # check if the low for the day is below stop loss level
            stop_loss_hit = (price_data['low'] < sim.breakeven_stop_loss_prices[stock])
            if stop_loss_hit:
                # calculate which price to use. some stocks gap down significantly
                stopped_out_price = min(sim.breakeven_stop_loss_prices[stock], price_data['open'])

                print(f'-> BREAKEVEN HIT ({stock}) @ ${stopped_out_price:.2f}')
                # add this to stops_hit
                stops_hit.append(dict(stock=stock, stopped_out_price=stopped_out_price, price_data=price_data))

    # Process exits for the stocks
    for hits in stops_hit:
        process_exit(sim, hits['stock'], hits['price_data'], forced_price=hits['stopped_out_price'])


#########################################
#                                       #
#         SIMULATION LOGIC 🔄           #
#                                       #
#########################################

def run_simulations_with_sampling(ws, start_date):
    reference_dates = get_reference_dates(start_date, ws)
    all_results = []
    all_simulations = {}

    for date in reference_dates:
        print(f"==> Running simulation starting from {date}")
        results_dict = {}
        simulations = {}

        modified_ws = data_filter_from_date(ws, date)

        # Randomly exclude rows based on the config parameter
        exclusion_rate = config["simulator"]["random_exclusion_rate"]
        modified_ws = randomly_exclude_rows(modified_ws, exclusion_rate)
        print(f"Randomly excluded {exclusion_rate:.1%} of rows for this sample.")

        for current_simultaneous_positions in config["simulator"]["simultaneous_positions"]:
            for take_profit_variant in config["simulator"]["take_profit_variants"]:
                for close_higher_percentage in config["simulator"]["close_higher_percentage_variants"]:
                    for stop_below_bullish_reference_variant in config["simulator"][
                        "stop_below_bullish_reference_variants"]:
                        variant_name = f"{current_simultaneous_positions}pos_{take_profit_variant['variant_name']}_chp{close_higher_percentage}_stp{stop_below_bullish_reference_variant}"
                        results_dict, latest_sim = run_simulation(
                            modified_ws, results_dict, take_profit_variant, close_higher_percentage,
                            stop_below_bullish_reference_variant,
                            current_simultaneous_positions
                        )
                        simulations[variant_name] = latest_sim

        all_results.append(results_dict)
        all_simulations[date] = simulations

    # Average the results
    averaged_results = average_dict_values(all_results)

    # Average the simulations
    averaged_simulations = {}
    for variant in all_simulations[reference_dates[0]].keys():
        averaged_simulations[variant] = Simulation(config["simulator"]["capital"])
        for attr in dir(all_simulations[reference_dates[0]][variant]):
            if not attr.startswith("__") and not callable(getattr(all_simulations[reference_dates[0]][variant], attr)):
                values = [getattr(all_simulations[date][variant], attr) for date in reference_dates]
                if isinstance(values[0], (int, float)):
                    setattr(averaged_simulations[variant], attr, statistics.mean(values))
                elif isinstance(values[0], dict):
                    setattr(averaged_simulations[variant], attr, average_dict_values(values))
                else:
                    setattr(averaged_simulations[variant], attr, values[0])

    return averaged_results, averaged_simulations, reference_dates

def run_simulation(ws, results_dict, take_profit_variant, close_higher_percentage, stop_below_bullish_reference_variant, current_simultaneous_positions):
    sim = Simulation(capital=config["simulator"]["capital"])
    sim.current_simultaneous_positions = current_simultaneous_positions

    print(f"Take profit variant {take_profit_variant['variant_name']} | "
          f"max positions {sim.current_simultaneous_positions} | close higher percentage variant {close_higher_percentage} | "
          f"stop below bullish candle variant {stop_below_bullish_reference_variant}")

    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)

    sim.balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
    sim.detailed_capital_values[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

    while current_date_dt < end_date_dt:
        date_changed_reported = False    # just to show info

        previous_date_month = current_date_dt.strftime("%m")
        current_date_dt = current_date_dt + timedelta(days=1)
        current_date_month = current_date_dt.strftime("%m")

        print(current_date_dt, "| positions: ", sim.current_positions, "| allocations: ", sim.entry_allocation)

        # Process pending stop loss updates and trail updates at the beginning of each day
        sim.process_pending_stop_loss_updates()
        sim.process_pending_trail_stop_updates()
        sim.process_pending_breakeven_stop_updates()

        if previous_date_month != current_date_month:
            sim.balances[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
        sim.detailed_capital_values[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

        # Entries
        day_entries = ws.loc[ws["entry_date"] == current_date_dt]
        for key, row in day_entries.iterrows():
            process_entry(sim, row["stock"], row["entry_price_actual"], take_profit_variant,
                          current_date_dt, row["initial_stop_loss"], close_higher_percentage,
                          stop_below_bullish_reference_variant)

        # Check if conditions for the second entry were met for applicable existing entries for all entered stocks
        # Here also move the stop after the second entry to a low of the bullish reference candle -X%
        for stock in sim.current_positions:

            price_data = get_price_from_db(stock, current_date_dt)  # get the prices
            if price_data['date_is_changed'] and not date_changed_reported:
                print(f"(i) using price date from {price_data['date']}, possibly weekend")
                date_changed_reported = True

            # Process for 2nd entry
            if sim.entry_allocation[stock] < 1:
                next_open_price = get_next_opening_price(stock, current_date_dt)
                sim.check_and_process_second_entry(stock, price_data['close'], next_open_price)

            # Process for the trailing stop
            if stock not in sim.trailing_stop_active:
                sim.check_and_update_trailing_stop(stock,
                                                    price_data['high'],
                                                    config["simulator"]["stop_loss_management"]["price_increase_trigger"],
                                                    config["simulator"]["stop_loss_management"]["new_stop_loss_level"]
                                                    )
            # Process for breakeven stop
            if config["simulator"]["breakeven_stop_loss"]["enabled"]:
                sim.check_and_update_breakeven_stop(stock,
                                                    price_data['high'],
                                                    config["simulator"]["breakeven_stop_loss"]["price_increase_trigger"]
                                                    )

        # Check whether stocks reach the profit level for each stock
        # Also check for stop losses
        if len(sim.current_positions) > 0:
            check_profit_levels(sim, current_date_dt, take_profit_variant)
            check_stop_loss(sim, current_date_dt)
            check_stop_breakeven(sim, current_date_dt)

            # Check whether fisher distance take profits should be triggered
            if config["simulator"]["fisher_distance_exit"]["enabled"]:
                check_fisher_based_take_profit(sim, current_date_dt, date_changed_reported)

        # Exits
        day_exits = ws.loc[ws[f"control_exit_date"] == current_date_dt]
        for key, row in day_exits.iterrows():
            price_data = get_price_from_db(row["stock"], current_date_dt)
            if price_data:
                process_exit(sim, row["stock"], price_data)

        sim.update_capital(sim.current_capital)  # Update capital values at the end of each day

    # Exit all remaining positions at the end of the simulation
    if len(sim.current_positions) > 0:
        print(f"[x] Stopped similation: exiting all remaining positions as of {end_date_dt}")
        exit_all_positions(sim, end_date_dt)

    # Add one more entry for the start of the next month
    next_month = (end_date_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    sim.balances[next_month.strftime("%d/%m/%Y")] = sim.current_capital

    # Calculate metrics and print the results
    sim.calculate_metrics()
    sim.print_metrics()

    # Saving the result in the overall dictionary
    # Update here if adding new iterations of variants!
    results_dict = update_results_dict(
        results_dict, sim, current_simultaneous_positions,
        take_profit_variant['variant_name'],
        close_higher_percentage,
        stop_below_bullish_reference_variant
    )
    return results_dict, sim


# Get and save stock prices if not available for running the simulation
def get_stock_prices(sheet_df, prices_start_date):
    stock_names = [item.stock for key, item in sheet_df.iterrows()]
    stock_markets = [item.market for key, item in sheet_df.iterrows()]

    if not arguments["forced_price_update"]:
        print(
            f"Skipping price updates in db, as there is no flag --forced_price_update")
        return {}

    # Convert prices_start_date to date if it's not already
    if isinstance(prices_start_date, str):
        prices_start_date = datetime.strptime(prices_start_date, '%Y-%m-%d').date()
    elif isinstance(prices_start_date, datetime):
        prices_start_date = prices_start_date.date()

    # Check if the Price table exists and get the earliest date
    earliest_date = check_earliest_price_date()

    # Ensure earliest_date is also a date object for comparison
    if earliest_date:
        if isinstance(earliest_date, datetime):
            earliest_date = earliest_date.date()
        elif isinstance(earliest_date, str):
            earliest_date = datetime.strptime(earliest_date, '%Y-%m-%d').date()
        ate_difference = abs((earliest_date - prices_start_date).days)
        # Can add logic here if need to manage update of prices

    # If the dates don't match or the table was just created, proceed with updating
    delete_all_prices()  # Clear existing data

    stock_prices = {}
    prices_to_add = []
    added_records = set()  # To keep track of unique (stock, date) combinations

    for i, stock in enumerate(stock_names):
        market = Market(stock_markets[i])
        print(f"Getting stock data for {stock}{market.stock_suffix}")
        stock_df = get_stock_data(f"{stock}{market.stock_suffix}", prices_start_date)
        stock_df = stock_df[0]  # this is the actual dataframe

        stock_prices[stock] = stock_df.to_dict('records')

        # Prepare data for bulk insert, avoiding duplicates
        for _, row in stock_df.iterrows():
            record_key = (stock, row['timestamp'])
            if record_key not in added_records:
                prices_to_add.append({
                    'stock': stock,
                    'date': row['timestamp'].to_pydatetime(),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
                added_records.add(record_key)

    # Bulk insert the new price data
    if prices_to_add:
        try:
            bulk_add_prices(prices_to_add)
        except IntegrityError as e:
            print(f"Error inserting prices: {e}")
            print("Some records may already exist in the database. Continuing with available data.")

    return stock_prices


#########################################
#                                       #
#        MAIN EXECUTION 🚀              #
#                                       #
#########################################
if __name__ == "__main__":

    # Get the run params
    arguments = define_simulator_args()

    print("reading the values...")

    # Dates for simulation
    start_date = arguments["start"]
    end_date = arguments["end"]

    # Price data start date is required in advance because we are calculating on a weekly scale
    prices_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(
        days=2 * 365
    )
    # ^^^ -2 years ago from start is ok for data handling
    prices_start_date = prices_start_date.strftime("%Y-%m-%d")

    # Get the sheet to a dataframe
    sheet_name = config["logging"]["gsheet_name"]
    tab_name = config["logging"]["gsheet_tab_name"]

    ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)
    ws.columns = config["logging"]["gsheet_columns"]
    ws = prepare_data(ws)

    # Dict to save all the results, start with empty
    results_dict = dict()

    # Dates filtering for the dataset in the sheet
    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)
    ws = data_filter_by_dates(ws, start_date_dt, end_date_dt)

    # Filter for a specific stock if provided
    if arguments["stock"]:
        ws = ws[ws['stock'] == arguments["stock"]]
        print(f"Filtered data for stock: {arguments['stock']}")
        if ws.empty:
            print(f"No data found for stock {arguments['stock']}. Exiting.")
            exit(0)

    # Filter the dataset per the config for the numerical parameters
    ws = filter_dataframe(ws, config)

    # Get information on the price data if the date is new
    get_stock_prices(ws, prices_start_date)

    if arguments["sampling"]:
        # Run simulations with sampling
        results_dict, simulations, reference_dates = run_simulations_with_sampling(ws, start_date)
        print(f"\n(i) Using the following start dates for sampling:")
        for reference_date in reference_dates:
            print(reference_date)
        print(f'(i) {config["simulator"]["random_exclusion_rate"]:.0%} of records were randomly excluded on each sample run')

    else:
        # Iterate over variants
        simulations = {}
        for current_simultaneous_positions in config["simulator"]["simultaneous_positions"]:
            for take_profit_variant in config["simulator"]["take_profit_variants"]:
                for close_higher_percentage in config["simulator"]["close_higher_percentage_variants"]:
                    for stop_below_bullish_reference_variant in config["simulator"]["stop_below_bullish_reference_variants"]:
                        # For logging
                        variant_name = f"{current_simultaneous_positions}pos_{take_profit_variant['variant_name']}_chp{close_higher_percentage}_stp{stop_below_bullish_reference_variant}"
                        # Run it
                        results_dict, latest_sim = run_simulation(
                            ws, results_dict, take_profit_variant, close_higher_percentage, stop_below_bullish_reference_variant, current_simultaneous_positions
                        )
                        simulations[variant_name] = latest_sim

    # Create the report
    create_report(results_dict, simulations, arguments["plot"])

