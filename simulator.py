#########################################
#                                       #
#         IMPORTS AND SETUP 🐍          #
#                                       #
#########################################

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

from peewee import IntegrityError

import libs.gsheetobj as gsheetsobj
from datetime import datetime, timedelta
from libs.stocktools import get_stock_data, Market
from libs.techanalysis import fisher_distance
from libs.stocktools import get_stock_data
from libs.db import get_historical_prices

import argparse
import pandas as pd

parser = argparse.ArgumentParser()

from libs.read_settings import read_config
config = read_config()

#from libs.signal import red_day_on_volume # not used
from libs.simulation import Simulation
from libs.db import check_earliest_price_date, delete_all_prices, bulk_add_prices, get_price_from_db
from libs.helpers import create_report, define_simulator_args, data_filter_by_dates, prepare_data, filter_dataframe

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

def process_entry(sim, stock, entry_price, take_profit_variant, current_date, initial_stop, close_higher_percentage, stop_below_bullish_reference_variant):

    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"(i) max possible positions | skipping {stock} entry")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        sim.capital_per_position[stock] = sim.current_capital / current_simultaneous_positions

        # Set fisher-related info for a stock
        sim.set_fisher_distance_profit_info(stock)

        # Set take profit levels
        sim.set_take_profit_levels(stock, take_profit_variant, entry_price)

        # Set initial entry and stop / allocation reference prices 
        allocation_reference_price = get_highest_body_level(stock, current_date)
        adjusted_stop_reference = get_lowest_price_before_entry(stock, current_date) * (1-stop_below_bullish_reference_variant)

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
    #TODO: wherever it is checked only take profit on fisher crossing if 5%+, also need to implement the logic of it going above 0.6 to be valid again
    #TODO: continue working on that, does not work correctly
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
        sim.update_trade_statistics(overall_result, current_simultaneous_positions)


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
    # Check if the date is ok. Otherwise would result in incorrect exit.
    if date_changed_reported:
        print('skipping fisher distance check because of the weekend or holiday')
        return

    reentry_threshold = config["simulator"]["fisher_distance_exit"]["reentry_threshold"]

    for stock in sim.current_positions:
        # Get historical data from the database
        historical_prices = get_historical_prices(stock, current_date_dt)

        # Check if we have new data since the last calculation
        last_price_date = historical_prices.index[-1]

        if (stock not in sim.last_fisher_calculation or
                sim.last_fisher_calculation[stock] is None or
                'date' not in sim.last_fisher_calculation[stock] or
                last_price_date > sim.last_fisher_calculation[stock]['date']):
            # Calculate Fisher distance only if we have new data
            fisher_df = fisher_distance(historical_prices)
            current_fisher_dist = fisher_df['distance'].iloc[-1]
            previous_fisher_dist = fisher_df['distance'].iloc[-2]

            # Store the new calculation
            sim.last_fisher_calculation[stock] = {
                'date': last_price_date,
                'current': current_fisher_dist,
                'previous': previous_fisher_dist
            }

        else:
            # Use stored values if no new data
            current_fisher_dist = sim.last_fisher_calculation[stock]['current']
            previous_fisher_dist = sim.last_fisher_calculation[stock]['previous']

        # Check if Fisher distance has gone above the reentry threshold
        if current_fisher_dist > reentry_threshold:
            sim.fisher_distance_above_threshold[stock] = True
            print(f"Fisher distance for {stock} went above reentry threshold: {current_fisher_dist:.4f}")

        # If crosses - need to exit the next day
        # Check for crossing zero downwards as well
        if (previous_fisher_dist > 0 and current_fisher_dist <= 0):
            if sim.fisher_distance_above_threshold[stock] and sim.fisher_distance_exits[stock]['number_exits'] < config["simulator"]["fisher_distance_exit"]["max_exits"]:
                print(f"-> Fisher distance for {stock} crossed below zero (from {previous_fisher_dist:.4f} to {current_fisher_dist:.4f})")
                next_day_dt = current_date_dt + timedelta(days=1)
                price_data = get_price_from_db(stock, next_day_dt)
                print(f"-- will take profit at the next day open price {price_data['open']}")
                sim.check_and_update_fisher_based_profit(stock, price_data['open'],
                                                         config["simulator"]["fisher_distance_exit"]["exit_proportion"],
                                                         config["simulator"]["commission"])
                sim.fisher_distance_above_threshold[stock] = False  # Reset the flag after taking profit
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


#########################################
#                                       #
#         SIMULATION LOGIC 🔄           #
#                                       #
#########################################

def run_simulation(results_dict, take_profit_variant, close_higher_percentage, stop_below_bullish_reference_variant):
    sim = Simulation(capital=config["simulator"]["capital"])
    print(f"Take profit variant {take_profit_variant['variant_name']} | "
          f"max positions {current_simultaneous_positions} | close higher percentage variant {close_higher_percentage} | "
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

        # Check whether stocks reach the profit level for each stock
        # Also check for stop losses
        if len(sim.current_positions) > 0:
            check_profit_levels(sim, current_date_dt, take_profit_variant)
            check_stop_loss(sim, current_date_dt)

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

    ### Uncomment for testing on a particular stock
    ws = ws[ws['stock'] == 'GOGL']

    # Filter the dataset per the config for the numerical parameters
    ws = filter_dataframe(ws, config)

    # Get information on the price data if the date is new
    get_stock_prices(ws, prices_start_date)

    # Iterate over take profit variants and number of positions
    simulations = {}

    for current_simultaneous_positions in config["simulator"]["simultaneous_positions"]:
        for take_profit_variant in config["simulator"]["take_profit_variants"]:
            for close_higher_percentage in config["simulator"]["close_higher_percentage_variants"]:
                for stop_below_bullish_reference_variant in config["simulator"]["stop_below_bullish_reference_variants"]:
                    # For logging
                    variant_name = f"{current_simultaneous_positions}pos_{take_profit_variant['variant_name']}_chp{close_higher_percentage}_stp{stop_below_bullish_reference_variant}"
                    # Run it
                    results_dict, latest_sim = run_simulation(
                        results_dict, take_profit_variant, close_higher_percentage, stop_below_bullish_reference_variant
                    )
                    simulations[variant_name] = latest_sim

    # Create the report
    create_report(results_dict, simulations, arguments["plot"])
