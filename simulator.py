# Simulates trading progress and results over time using a spreadsheet with various considerationssimulator_result.csv

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

from peewee import IntegrityError

import libs.gsheetobj as gsheetsobj
#from libs.signal import red_day_on_volume #not used
from datetime import datetime, timedelta
from libs.stocktools import get_stock_data, Market
import argparse

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, numbers

# For plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openpyxl.drawing.image import Image
import io

parser = argparse.ArgumentParser()

from libs.read_settings import read_config
config = read_config()

from libs.simulation import Simulation
from libs.db import check_earliest_price_date, delete_all_prices, bulk_add_prices, get_price_from_db

pd.set_option("display.max_columns", None)

def create_variant_plot(sim, variant_name):
    # Convert dates to datetime objects
    dates = [datetime.strptime(date, "%d/%m/%Y") for date in sim.detailed_capital_values.keys()]
    values = list(sim.detailed_capital_values.values())

    # Add termination value as a step change
    last_date = dates[-1]
    next_month = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)

    plt.figure(figsize=(12, 6))

    # Plot the main line
    plt.step(dates, values, where='post')

    # Add the final step
    plt.step([last_date, next_month], [values[-1], sim.current_capital], where='post')

    plt.title(f"Capital Over Time - {variant_name}")
    plt.xlabel("Date")
    plt.ylabel("Capital")

    # Set x-axis to show only the first of each month
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%01'))

    # Extend x-axis slightly to show the final step
    plt.xlim(dates[0], next_month + timedelta(days=1))

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf

def adjust_column_width(worksheet):
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column].width = adjusted_width

def set_font_size(worksheet, size):
    for row in worksheet.iter_rows():
        for cell in row:
            cell.font = Font(size=size)

def format_percentage(value):
    return f"{value:.2%}"

def format_number(value):
    return f"{value:.2f}"

def define_args():
    # Take profit levels variation is only supported for the control group, thus the modes are different
    # Removing this as not used now, only one mode using the sheet
    # parser.add_argument(
    #     "-mode",
    #     type=str,
    #     required=True,
    #     help="Mode to run the simulation in (main|tp). Main mode means taking profit as per the spreadsheet setup.",
    #     choices=["main", "tp"],
    # )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the latest simulation"
    )
    parser.add_argument(
        "--failsafe", action="store_true", help="Activate the failsafe approach"
    )
    parser.add_argument(
        "--forced_price_update", action="store_true", help="Activate the failsafe approach"
    )
    #
    # parser.add_argument(
    #     "--show_monthly", action="store_true", help="Show MoM capital value (only in main mode)"
    # )

    parser.add_argument(
        "-method",
        type=str,
        required=True,
        choices=["mri", "anx"],
        help="Method of shortlisting (mri or anx)"
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

    # Not used
    # # Arguments to overwrite default settings for filtering
    # parser.add_argument(
    #     "--red_day_exit",
    #     action="store_true",
    #     help="Exit when entry day is red (in tp mode only)",
    # )

    # Not used
    # Test for the exit price variation experimentation
    # parser.add_argument(
    #     "--exit_variation_a",
    #     action="store_true",
    #     help="Exit approach experiment A (main mode only)",
    # )

    args = parser.parse_args()
    arguments = vars(args)

    # Convert specific arguments to boolean, defaulting to False if not provided
    boolean_args = ["plot", "failsafe", "forced_price_update"]   # "show_monthly"
    arguments.update({arg: bool(arguments.get(arg)) for arg in boolean_args})

    return arguments


# def process_filter_args():
#     red_entry_day_exit = True if arguments["red_day_exit"] else False
#     failsafe = True if arguments["failsafe"] else False
#     return red_entry_day_exit, failsafe


def p2f(s):
    try:
        stripped_s = s.strip("%")
        if stripped_s == "":
            return
        else:
            return float(stripped_s) / 100
    except AttributeError:
        return s


def data_filter_by_dates(ws, start_date, end_date):
    ws = ws.loc[ws["entry_date"] >= start_date]
    ws = ws.loc[ws["entry_date"] <= end_date]
    return ws


def prepare_data(ws):
    # Convert types
    # This should be in gsheetobj
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
        "weekly_mri_count",
        "fisher_daily",
        "fisher_weekly",
        "coppock_daily",
        "coppock_weekly"
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


def get_lowest_price_before_entry(stock, entry_date):
    previous_date = entry_date - timedelta(days=1)
    price_info = get_price_from_db(stock, previous_date)
    return price_info['low']


def process_entry(sim, stock, entry_price, take_profit_variant, current_date):
    sim.set_take_profit_levels(stock, take_profit_variant)

    if len(sim.current_positions) + 1 > current_simultaneous_positions:
        print(f"max possible positions held, skipping {stock}")
    else:
        sim.positions_held += 1
        sim.current_positions.add(stock)
        sim.capital_per_position[stock] = sim.current_capital / current_simultaneous_positions
        sim.entry_prices[stock] = entry_price
        sim.set_take_profit_levels(stock, take_profit_variant)

        # Calculate and set stop loss price
        lowest_price_before_entry = get_lowest_price_before_entry(stock, current_date)

        stop_loss_price = lowest_price_before_entry * (1 - config["simulator"]["stop_loss_level"])
        sim.set_stop_loss(stock, stop_loss_price)

        sim.current_capital -= config["simulator"]["commission"]

        # Show info
        print(f"-> ENTER {stock} | positions held: {sim.positions_held}")
        print(f'-- commission ${config["simulator"]["commission"]}')
        print(f"-- stop loss set @ ${stop_loss_price:.2f}")
        print(f"-- current capital on entry: ${sim.current_capital}, allocated to the position: ${sim.capital_per_position[stock]}")


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

def process_exit(sim, stock_code, price_data, forced_price=None):
    if stock_code in sim.current_positions:

        entry_price = sim.entry_prices[stock_code]
        total_position_size = sim.capital_per_position[stock_code]

        # Check if we need to use specific price
        exit_price = forced_price if forced_price is not None else price_data['open']

        final_exit_proportion = 1 - sim.take_profit_info[stock_code]['taken_profit_proportion']
        final_price_change = (exit_price - entry_price) / entry_price

        print(f"-> Full exit ({stock_code}) exit price: ${exit_price:.2f} | entry ${entry_price:.2f} | change {final_price_change:.2%}")

        # Calculate the contribution from the take profit levels
        # This is representing the growth from the overall original amount, accounted for the proportion of TP levels
        tp_contribution = calculate_profit_contribution(sim.take_profit_info[stock_code])
        print(f'-- take profit contribution {tp_contribution:.2%}')

        # Calculate the final exit part contribution
        last_exit_contribution = final_price_change*final_exit_proportion
        print(f'-- final exit contribution {last_exit_contribution:.2%}')

        overall_result = tp_contribution + last_exit_contribution
        print(f'-- overall result {overall_result:.2%}')

        # Calculate the outcome using the original position size
        profit_amount = total_position_size * (overall_result)
        print(f'--> profit/loss ${profit_amount:.2f}')

        previous_capital = sim.current_capital
        sim.current_capital += profit_amount
        sim.current_capital -= config["simulator"]["commission"]
        print(f"Capital {previous_capital} -> {sim.current_capital}")

        # Delete traces
        sim.current_positions.remove(stock_code)
        sim.positions_held -= 1
        del sim.capital_per_position[stock_code]
        del sim.take_profit_info[stock_code]
        sim.trailing_stop_active.pop(stock_code, None)

        # Results update
        sim.update_capital(sim.current_capital)
        sim.update_trade_statistics(overall_result, current_simultaneous_positions)


def update_results_dict(
    results_dict,
    sim,
    current_simultaneous_positions,
    take_profit_variant_name,
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
        simultaneous_positions=current_simultaneous_positions,
        variant_group=current_variant,
    )
    results_dict[
        f"{current_variant}_{current_simultaneous_positions}pos_{take_profit_variant_name}{extra_suffix}"
    ] = result_current_dict
    return results_dict


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


def run_simulation(results_dict, take_profit_variant):
    sim = Simulation(capital=config["simulator"]["capital"])
    print(f"Take profit variant {take_profit_variant['variant_name']} | max positions {current_simultaneous_positions}")

    start_date_dt, end_date_dt, current_date_dt = get_dates(start_date, end_date)

    sim.balances[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
    sim.detailed_capital_values[start_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

    while current_date_dt < end_date_dt:
        previous_date_month = current_date_dt.strftime("%m")
        current_date_dt = current_date_dt + timedelta(days=1)
        current_date_month = current_date_dt.strftime("%m")

        print(current_date_dt, "| positions: ", sim.current_positions)

        # Process pending stop loss updates at the beginning of each day
        sim.process_pending_stop_loss_updates()

        if previous_date_month != current_date_month:
            sim.balances[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital
        sim.detailed_capital_values[current_date_dt.strftime("%d/%m/%Y")] = sim.current_capital

        # Entries
        day_entries = ws.loc[ws["entry_date"] == current_date_dt]
        for key, row in day_entries.iterrows():
            process_entry(sim, row["stock"], row["entry_price_actual"], take_profit_variant, current_date_dt)

        # Check whether stocks reach the profit level for each stock
        # Also check for stop losses
        if len(sim.current_positions) > 0:
            check_profit_levels(sim, current_date_dt, take_profit_variant)
            check_stop_loss(sim, current_date_dt)

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
    results_dict = update_results_dict(
        results_dict, sim, current_simultaneous_positions, take_profit_variant['variant_name']
    )
    return results_dict, sim

def create_monthly_breakdown(simulations):
    monthly_data = {}
    for variant, sim in simulations.items():
        for date, value in sim.balances.items():
            date_obj = datetime.strptime(date, "%d/%m/%Y")
            month_start = date_obj.strftime("%d/%m/%Y")  # Keep the day as-is
            if month_start not in monthly_data:
                monthly_data[month_start] = {}
            monthly_data[month_start][variant] = value

    df = pd.DataFrame(monthly_data).T
    df.index.name = 'Date'
    df.reset_index(inplace=True)
    return df

# Apply filters from config to the DataFrame
def filter_dataframe(df, config):
    filters = config["simulator"]["numerical_filters"]
    for column, conditions in filters.items():
        if isinstance(conditions, dict):  # Ensure conditions is a dictionary
            if 'min' in conditions:
                df = df[df[column] >= conditions['min']]
            if 'max' in conditions:
                df = df[df[column] <= conditions['max']]
        else:
            print(f"Error: Filter conditions for {column} are not specified correctly.")
    return df


# Get and save stock prices if not available for running the simulation
def get_stock_prices(sheet_df, prices_start_date):
    stock_names = [item.stock for key, item in sheet_df.iterrows()]
    stock_markets = [item.market for key, item in sheet_df.iterrows()]

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

        # Check if the earliest date in the database is within 5 days of the start date
        date_difference = abs((earliest_date - prices_start_date).days)
        if (date_difference <= 5) and not arguments["forced_price_update"]:
            print(f"Data within 5 days of {prices_start_date} already exists in the database. Skipping update.")
            return {}

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


############## MAIN CODE ###############
if __name__ == "__main__":

    arguments = define_args()

    #red_entry_day_exit, failsafe = process_filter_args()

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

    ### TEST TEST
    #ws = ws[ws['stock'] == 'AGIO']

    # Filter the dataset per the config for the numerical parameters
    ws = filter_dataframe(ws, config)

    # Get information on the price data if the date is new
    get_stock_prices(ws, prices_start_date)

    # > Iterate over take profit variants and number of positions
    simulations = {}

    for current_simultaneous_positions in config["simulator"]["simultaneous_positions"]:
        for take_profit_variant in config["simulator"]["take_profit_variants"]:
            # For logging
            variant_name = f"{current_simultaneous_positions}pos_{take_profit_variant['variant_name']}"
            # Run it
            results_dict, latest_sim = run_simulation(
                results_dict, take_profit_variant
            )
            simulations[variant_name] = latest_sim

    # < Stop iterations

    #### Finalisation
    # Write the output to a dataframe and a spreadsheet
    resulting_dataframes = []
    for k, v in results_dict.items():
        print(k, v)
        values_current = v.copy()
        values_current["variant"] = k
        resulting_dataframes.append(
            pd.DataFrame.from_records(values_current, index=[0])
        )

    # Create the summary DataFrame
    final_result = pd.concat(pd.DataFrame.from_records(v, index=[0]) for v in results_dict.values())
    final_result['variant'] = results_dict.keys()
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
            "average_mom_growth",
            "winning_trades_number",
            "worst_trade_adjusted",
        ]
    ]

    # Format percentage and number columns
    percentage_cols = ["best_trade_adjusted", "growth", "max_drawdown", "win_rate", "median_mom_growth", "average_mom_growth", "worst_trade_adjusted"]
    number_cols = ["losing_trades_number", "max_negative_strike", "winning_trades_number"]

    final_result[percentage_cols] = final_result[percentage_cols].applymap(format_percentage)
    final_result[number_cols] = final_result[number_cols].applymap(format_number)

    # Create the monthly breakdown DataFrame
    monthly_breakdown = create_monthly_breakdown(simulations)

    # Create the monthly breakdown DataFrame
    monthly_breakdown = create_monthly_breakdown(simulations)
    monthly_breakdown['Date'] = pd.to_datetime(monthly_breakdown['Date'], format='%d/%m/%Y')
    monthly_breakdown = monthly_breakdown.sort_values('Date')
    monthly_breakdown['Date'] = monthly_breakdown['Date'].dt.strftime('%d/%m/%Y')

    # Format monthly breakdown values
    monthly_breakdown.iloc[:, 1:] = monthly_breakdown.iloc[:, 1:].applymap(format_number)

    # Create Excel file with three sheets
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Summary"

    for r in dataframe_to_rows(final_result, index=False, header=True):
        ws1.append(r)

    ws2 = wb.create_sheet(title="Monthly Breakdown")
    for r in dataframe_to_rows(monthly_breakdown, index=False, header=True):
        ws2.append(r)

    # Adjust column width and set font size for both sheets
    for ws in [ws1, ws2]:
        adjust_column_width(ws)
        set_font_size(ws, 12)

    # Add plots if the plot argument is provided
    if arguments["plot"]:
        ws3 = wb.create_sheet(title="Plots")
        row = 1
        for variant_name, sim in simulations.items():
            img_buf = create_variant_plot(sim, variant_name)
            img = Image(img_buf)
            img.width = 900
            img.height = 500
            ws3.add_image(img, f'A{row}')

            row += 30  # Spacing between plots

    # Save the Excel file
    excel_filename = "sim_summary.xlsx"
    wb.save(excel_filename)
    print(f"(i) Detailed info saved to {excel_filename}")
