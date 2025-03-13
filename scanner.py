# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")
from collections import namedtuple
from collections import defaultdict
import peewee
from tqdm import tqdm

# For concurrent fetching of stock prices
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from libs.helpers import (
    define_scanner_args,
    dates_diff,
    format_number,
    get_previous_workday,
    get_current_workday,
    get_previous_workday_from_date,
    get_test_stocks,
    get_current_and_lookback_date,
    get_data_start_date,
    create_header
)
from libs.signal import (
    bullish_mri_based,
    market_bearish,
    bullish_anx_based,
    earnings_gap_down,
    bearish_anx_based,
    earnings_gap_down_in_range
)
from libs.stocktools import (
    get_stock_data,
    ohlc_daily_to_weekly,
    get_exchange_symbols,
    get_earnings_calendar,
    Market
)
from libs.db import (
    bulk_add_stocks,
    create_stock_table,
    delete_all_stocks,
    get_stocks,
    get_update_date,
    delete_all_stock_prices,
    create_stock_price_table,
    bulk_add_stock_prices,
    get_stock_price_data,
    initialize_price_database
)
from libs.techanalysis import td_indicators, MA, fisher_distance, coppock_curve
import pandas as pd
from time import time, sleep
from datetime import datetime, timedelta

from libs.read_settings import read_config
config = read_config()


def rewrite_stocks(exchange, stocks):
    create_stock_table()
    print(f"Deleting the existing stocks for {exchange}")
    delete_all_stocks(exchange)
    print(f"Writing info on {len(stocks)} stocks to the database")
    bulk_add_stocks(stocks)
    print("Update finished")


def get_current_date():
    if arguments["date"] is None:
        if not config["locality"]["shift_update_day"]:
            current_date = get_current_workday()
        else:
            current_date = get_previous_workday()
            #print(f'(i) Shifting update date to previous workday per config')
    else:
        current_date = arguments["date"].strftime("%Y-%m-%d")

    return current_date


def update_stocks(active_markets):
    checked_workday = get_current_date()
    print(f"Updating info on traded stocks as of {checked_workday}")

    for market in active_markets:
        print(f'Updating stock list for {market.market_code}')
        stocks = get_exchange_symbols(market, checked_workday, config["filters"]["minimum_market_cap"])
        rewrite_stocks(market.market_code, stocks)

def check_update_date(active_markets):

    for market in active_markets:
        try:
            last_update_date = get_update_date(market.market_code)
            diff = dates_diff(last_update_date)
            if diff > 1:
                print(
                    "Warning: Stocks list was not updated today, the volume filter could work incorrectly. "
                    "Please consider running the --update first..."
                )
                sleep(3)
        except:
            print(
                "Warning: failed to check the update date for correctness"
            )

def last_volume_5D_MA(volume_daily):
    volume_ma_20 = MA(volume_daily, 20, colname="volume")
    return volume_ma_20["ma20"].iloc[-1]


def generate_indicators_daily_weekly(ohlc_daily):
    # Generates extra info from daily OHLC
    if len(ohlc_daily) < 8:
        print("Too recent asset, not enough daily data")
        return None, None
    else:
        td_values = td_indicators(ohlc_daily)
        ohlc_with_indicators_daily = pd.concat([ohlc_daily, td_values], axis=1)

    ohlc_weekly = ohlc_daily_to_weekly(ohlc_daily)
    if len(ohlc_weekly) < 8:
        print("Too recent asset, not enough weekly data")
        return None, None
    else:
        td_values_weekly = td_indicators(ohlc_weekly)
        ohlc_with_indicators_weekly = pd.concat([ohlc_weekly, td_values_weekly], axis=1)

    return ohlc_with_indicators_daily, ohlc_with_indicators_weekly


def report_on_shortlist(market_code, direction, shortlist, exchange):
    if direction.upper() == 'BULL':
        direction_description = 'BULL 💹'
    elif direction.upper() == 'BEAR':
        direction_description = 'BEAR 🔻'

    print()
    if len(shortlist) > 0:
        print(create_header(f"Results for {market_code} ({direction_description})"))
        checked_workday = get_current_date()
        print(f"{len(shortlist)} shortlisted stocks on {checked_workday}:")

        # Group stocks by note type
        groups = {}
        for stock in shortlist:
            note = stock.note if stock.note else ""
            if note not in groups:
                groups[note] = []
            groups[note].append(stock)

        # Sort each group by volume
        for note, stocks in groups.items():
            stocks.sort(key=lambda x: x.volume, reverse=True)

            # Print header for each group
            print(f"\nShortlist {note}")
            for stock in stocks:
                print(f"{stock.code} ({stock.name}) | Volume {stock.volume}")

                # Unnecessary
                '''
                if hasattr(stock, 'green_star_info') and stock.green_star_info:
                    gs = stock.green_star_info
                    print(f"  └─ Green Star: TD{gs['current_td_setup']} | "
                          f"Close: ${gs['current_close']:.2f} | "
                          f"Prev: ${gs['previous_close']:.2f} | "
                          f"TD1: ${gs['td1_close']:.2f}")
                '''
    else:
        print(create_header(f"No shortlisted stocks for {market_code} ({direction_description})"))

def report_on_sentiment(shortlists):
    # Report on market sentiment when both directions are in the config
    configured_directions = config["strategy"][arguments["method"]]['directions']
    if ('bull' in configured_directions and 'bear' in configured_directions):
        total_bull = sum(len(shortlists[market.market_code]['bull'])
                         for market in active_markets
                         if 'bull' in shortlists[market.market_code])

        total_bear = sum(len(shortlists[market.market_code]['bear'])
                         for market in active_markets
                         if 'bear' in shortlists[market.market_code])

        # Print summary totals and sentiment
        sentiment = "Bearish 🐻" if total_bear > total_bull else "Bullish 🐂"

        print(create_header("Market Sentiment"))
        print(f"{sentiment} ({total_bull} bullish | {total_bear} bearish)")


def process_data_at_date(ohlc_daily, volume_daily):
    # Removes most recent columns if there is an argument to look at a particular date
    # < in the condition because we assume that at a day we only have info on the previous day close
    if arguments["date"] is None:
        return ohlc_daily, volume_daily

    ohlc_daily_shifted = ohlc_daily[ohlc_daily["timestamp"] < arguments["date"]]
    volume_daily_shifted = volume_daily[volume_daily["timestamp"] < arguments["date"]]

    return ohlc_daily_shifted, volume_daily_shifted


def process_market_data_at_date(market_ohlc_daily, market_volume_daily):
    if arguments["date"] is None:
        return market_ohlc_daily, market_volume_daily

    market_ohlc_daily_shifted = market_ohlc_daily[market_ohlc_daily["timestamp"] < arguments["date"]]
    market_volume_daily_shifted = market_volume_daily[market_volume_daily["timestamp"] < arguments["date"]]

    return market_ohlc_daily_shifted, market_volume_daily_shifted


def calculate_extra_metrics(ohlc_with_indicators_daily, ohlc_with_indicators_weekly):
        metric_values = dict()

        metric_values['fisherDaily'] = fisher_distance(ohlc_with_indicators_daily).iloc[-1].values[0]
        metric_values['fisherWeekly'] = fisher_distance(ohlc_with_indicators_weekly).iloc[-1].values[0]
        metric_values['coppockDaily'] = coppock_curve(ohlc_with_indicators_daily).iloc[-1].values[0]
        metric_values['coppockWeekly'] = coppock_curve(ohlc_with_indicators_weekly).iloc[-1].values[0]

        return metric_values


def scan_stock(stocks, market, method, direction, start_date):
    # Scans the stocks using particular method and strategy

    stock_suffix = market.stock_suffix
    shortlisted_stocks = []
    # Placeholder for shortlisted stocks and their attributes
    # Each stock will be a named tuple with the following definition:
    Stock = namedtuple('Stock', ['code', 'name', 'volume', 'note', 'green_star_info'])

    # Iterate through the list of stocks
    for i, stock in enumerate(stocks):
        print(f"\n{stock.code} [{stock.name}] ({i + 1}/{len(stocks)})")

        # Obtain OHLC data for the stocks
        # Get data from local database instead of API
        ohlc_daily, volume_daily = get_stock_price_data(stock.code, start_date)
        if ohlc_daily is None:
            print("No data available for the asset")
            continue

        ohlc_daily, volume_daily = process_data_at_date(ohlc_daily, volume_daily)

        (
            ohlc_with_indicators_daily,
            ohlc_with_indicators_weekly,
        ) = generate_indicators_daily_weekly(ohlc_daily)
        if (
            ohlc_with_indicators_daily is None
            or ohlc_with_indicators_weekly is None
        ):
            continue

        # Check for confirmation depending on the method
        if method == 'mri':
            # Raise NotImplemented because directional scan is not supported
            raise NotImplementedError("Directional scan not supported for MRI method")
            """
            confirmation, _ = bullish_mri_based(
                ohlc_with_indicators_daily,
                volume_daily,
                ohlc_with_indicators_weekly,
                consider_volume_spike=True,
                output=True,
                stock_name=stock.name,
            )
            """
        elif method == 'anx':
            if direction == 'bull':
                confirmation, numerical_score, trigger_note = bullish_anx_based(
                    ohlc_with_indicators_daily,
                    volume_daily,
                    ohlc_with_indicators_weekly,
                    output=True,
                    stock_name=stock.name,
                )
            elif direction == 'bear':
                confirmation, numerical_score, trigger_note = bearish_anx_based(
                    ohlc_with_indicators_daily,
                    volume_daily,
                    ohlc_with_indicators_weekly,
                    output=True,
                    stock_name=stock.name,
                )
        elif method == 'earnings':
            confirmation, trigger_note, gap_info, green_star_info = check_earnings_green_star(
                stock, market, ohlc_with_indicators_daily, volume_daily,
                ohlc_with_indicators_weekly, start_date
            )


        if confirmation:
            print(f"{stock.name} [v] meeting shortlisting conditions")
            volume_MA_5D = last_volume_5D_MA(volume_daily)

            if volume_MA_5D > config["filters"]["minimum_volume_level"]:
                print(
                    f'\n{stock.name} [v] meeting minimum volume level conditions '
                    f'({format_number(volume_MA_5D)} > {format_number(config["filters"]["minimum_volume_level"])})'
                )
                # Calculate extra metrics only for shortlisted stocks for a faster process
                # metric_data = calculate_extra_metrics(ohlc_with_indicators_daily, ohlc_with_indicators_weekly)

                # Append the shortlist with a stock and its characteristics
                shortlisted_stocks.append(
                    Stock(code=stock.code,
                          name=stock.name,
                          volume=volume_MA_5D,
                          note=trigger_note,
                          green_star_info=green_star_info
                          )
                )

            else:
                print(
                    f'\n{stock.name} [x] not meeting minimum volume level conditions '
                    f'({format_number(volume_MA_5D)} < {format_number(config["filters"]["minimum_volume_level"])})'
                )

        else:
            print(f"\n{stock.name} [x] not meeting shortlisting conditions")

    return shortlisted_stocks


def get_stocks_to_scan(market, method):
    """
    Get stocks for scanning based on method

    Args:
        market: Market object with market parameters
        method: Scanning method (mri, anx, earnings)

    Returns:
        List of stocks to scan
    """
    global current_date, lookback_date

    if method == 'earnings':
        # Get earnings stocks from StockTwits API
        earnings_stocks = get_earnings_calendar(lookback_date, current_date)

        if earnings_stocks:
            # Get stocks from database with our standard filters
            db_stocks = get_stocks(
                exchange=market.market_code,
                price_min=config["pricing"]["min"],
                price_max=config["pricing"]["max"],
                min_volume=config["filters"]["minimum_volume_level"],
            )

            # Filter to only stocks that have earnings
            filtered_stocks = [
                stock for stock in db_stocks
                if stock.code in earnings_stocks
            ]

            print(f"Found {len(filtered_stocks)} stocks in database with earnings that meet earning date criteria")
            return filtered_stocks

        return []

    else:
        # Default behavior for other methods
        return get_stocks(
            exchange=market.market_code,
            price_min=config["pricing"]["min"],
            price_max=config["pricing"]["max"],
            min_volume=config["filters"]["minimum_volume_level"],
        )


def scan_exchange_stocks(market, method, direction):
    """
    Function to scan the market for relevant stocks
    """

    """
    # Check the market conditions: not using it
    market_ohlc_daily, market_volume_daily = get_stock_data(market.related_market_ticker, reporting_date_start)
    market_ohlc_daily, market_volume_daily = process_market_data_at_date(market_ohlc_daily, market_volume_daily)
    is_market_bearish, _ = market_bearish(market_ohlc_daily, market_volume_daily, output=True)

    if is_market_bearish:
        print("Overall market sentiment is bearish, not scanning individual stocks")
        exit(0)
    """

    # Get the stocks for scanning
    if arguments["stocks"] is None:
        stocks = get_stocks_to_scan(market, method)
    else:
        # Pass the parameter
        stocks = get_stocks(
            codes=arguments["stocks"]
        )

    # Limit per arguments as required
    if arguments["num"] is not None:
        print(f"Limiting to the first {arguments['num']} stocks")
        stocks = stocks[: arguments["num"]]

    total_number = len(stocks)
    print(
        f'Scanning {total_number} stocks priced {config["pricing"]["min"]} from to {config["pricing"]["max"]} '
        f'and with volume of at least {format_number(config["filters"]["minimum_volume_level"])}\n'
    )

    # Fetch and store stock data for all stocks
    start_date = get_data_start_date(arguments["date"])
    fetch_and_store_stock_data(stocks, start_date)

    shortlist = scan_stock(stocks, market, method, direction, start_date)

    # Sort the list by volume in decreasing order
    sorted_stocks = sorted(shortlist, key=lambda stock: stock.volume, reverse=True)

    return sorted_stocks


def check_green_star_for_stock(stock_code, market, ohlc_daily):
    """
    Check if a stock meets the green star pattern criteria
    """
    try:
        if len(ohlc_daily) < 10:  # Need enough data for TD setup
            print(f"Insufficient data for {stock_code}")
            return False, None

        # Calculate TD indicators
        td_values = td_indicators(ohlc_daily)

        # Get latest values
        current_td_direction = td_values['td_direction'].iloc[-1]
        current_td_setup = td_values['td_setup'].iloc[-1]
        current_close = ohlc_daily['close'].iloc[-1]

        # Only proceed if we're in a green TD sequence
        if current_td_direction != 'green' or current_td_setup == 0:
            return False, None

        # Find the TD1 candle of the current sequence by looking backwards
        td1_index = None
        sequence_start_found = False
        pattern_already_triggered = False

        for i in range(len(td_values) - 2, -1, -1):
            # Break if we hit a non-green candle (end of current sequence)
            if td_values['td_direction'].iloc[i] != 'green':
                break

            # If we find TD1, mark its position
            if td_values['td_setup'].iloc[i] == 1:
                td1_index = i
                sequence_start_found = True
                break

        if not sequence_start_found:
            return False, None

        # Now check if pattern already triggered in this sequence
        # Look from TD1 to current position
        td1_close = ohlc_daily['close'].iloc[td1_index]

        for i in range(td1_index + 2, len(td_values) - 1):  # Start 2 candles after TD1
            if td_values['td_direction'].iloc[i] != 'green':
                break

            check_close = ohlc_daily['close'].iloc[i]
            check_prev_close = ohlc_daily['close'].iloc[i - 1]

            # If we find a previous trigger in this sequence, mark it
            if check_close > td1_close and check_close > check_prev_close:
                pattern_already_triggered = True
                break

        # Skip if pattern already triggered in this sequence
        if pattern_already_triggered:
            return False, None

        # Check current candle for pattern
        previous_close = ohlc_daily['close'].iloc[-2]

        if (current_close > td1_close and
                current_close > previous_close and
                td_values['td_direction'].iloc[-2] == 'green'):  # Verify previous candle was also green

            # Return relevant information about the green star pattern
            pattern_info = {
                'current_close': current_close,
                'previous_close': previous_close,
                'td1_close': td1_close,
                'current_td_setup': current_td_setup
            }

            return True, pattern_info

        return False, None
    except Exception as e:
        print(f"Error processing green star check for {stock_code}: {e}")
        return False, None


def check_earnings_green_star(stock, market, ohlc_daily, volume_daily, ohlc_weekly, start_date):
    """
    Check if a stock has both earnings gap down and green star pattern
    """
    # Get configuration settings
    lookback_period = config["strategy"].get("earnings", {}).get("lookback_period_days", 14)
    check_green_star = config["strategy"].get("earnings", {}).get("green_star_check", True)
    require_green_star = config["strategy"].get("earnings", {}).get("require_green_star", False)

    # Check for earnings gap down
    gap_confirmation, gap_info = earnings_gap_down_in_range(
        ohlc_daily,
        volume_daily,
        ohlc_weekly,
        lookback_days=lookback_period,
        output=True,
        stock_name=stock.name,
    )

    if not gap_confirmation:
        return False, "", None, None

    # Form initial trigger note
    trigger_note = ''
    '''
    if hasattr(gap_info, 'get') and gap_info.get('date') is not None:
        trigger_note = f"Gap down on {gap_info['date']} ({gap_info['gap_percent']:.1%})"
    else:
        trigger_note = f"Gap down detected ({gap_info.get('gap_percent', 0):.1%})"
    '''

    # If green star check is not enabled, return now
    if not check_green_star:
        return True, trigger_note, gap_info, None

    # Check for green star pattern (pass the ohlc_daily directly)
    green_star_found, green_star_info = check_green_star_for_stock(
        stock.code, market, ohlc_daily
    )

    if green_star_found:
        trigger_note += " | Green Star Pattern"
        print(f"-> Green Star Pattern detected for {stock.name}")
        print(f"   TD setup: {green_star_info['current_td_setup']}")
        print(f"   Current: ${green_star_info['current_close']:.2f}")
        print(f"   Previous: ${green_star_info['previous_close']:.2f}")
        print(f"   TD1: ${green_star_info['td1_close']:.2f}")

        return True, trigger_note, gap_info, green_star_info

    # If green star is required but not found, return failure
    if require_green_star:
        print(f"-> No Green Star Pattern found for {stock.name} - excluding from shortlist")
        return False, "", None, None

    # Otherwise return success but with no green star info
    return True, trigger_note, gap_info, None




def scan_stocks(active_markets):
    # Create shortlists placeholder for each market
    shortlists = defaultdict(lambda: defaultdict(list))

    if 'directions' not in config["strategy"][arguments["method"]].keys():
        print('Error: Directions for the strategy must be specified in the config')
        exit(0)

    # Initialize price database unless using existing data
    if not arguments["use_existing_price_data"]:
        initialize_price_database()

    # First pass: get all stocks and fetch data
    start_date = get_data_start_date(arguments["date"])
    all_market_stocks = {}
    processed_stocks = set()  # Keep track of stocks we've already processed

    for market in active_markets:
        print(f"\nProcessing {market.market_code}...")
        if arguments["stocks"] is None:
            stocks = get_stocks_to_scan(market, arguments["method"])
        else:
            stocks = get_stocks(codes=arguments["stocks"])

        if arguments["num"] is not None:
            print(f"Limiting to the first {arguments['num']} stocks")
            stocks = stocks[: arguments["num"]]

        # Filter out already processed stocks
        stocks_to_process = []
        for stock in stocks:
            if stock.code not in processed_stocks:
                stocks_to_process.append(stock)
                processed_stocks.add(stock.code)

        all_market_stocks[market.market_code] = stocks

        if stocks_to_process:
            total_number = len(stocks_to_process)
            print(
                f'Processing {total_number} stocks priced {config["pricing"]["min"]} to {config["pricing"]["max"]} '
                f'and with volume of at least {format_number(config["filters"]["minimum_volume_level"])}\n'
            )

            # Fetch and store data for this market's new stocks
            fetch_and_store_stock_data(stocks_to_process, start_date)
        else:
            print("All stocks already processed, skipping data fetch")

    # Second pass: run scans for each market using stored data
    for market in active_markets:
        stocks = all_market_stocks[market.market_code]
        for direction in config["strategy"][arguments["method"]]['directions']:
            print(f"\nScanning {market.market_code} for {direction.upper()} signals...")
            shortlists[market.market_code][direction] = scan_stock(stocks, market, arguments["method"], direction,
                                                                   start_date)

    # Report results
    print("\nFinished scanning")
    print()
    report_on_sentiment(shortlists)

    for market in active_markets:
        for direction in config["strategy"][arguments["method"]]['directions']:
            report_on_shortlist(
                market.market_code,
                direction,
                shortlists[market.market_code][direction],
                market.market_code,
            )


def fetch_prices_for_stock(stock, market, start_date):
    """
    Fetch price data for a single stock.

    Args:
        stock: Stock object containing code and exchange info
        market: Market object for the stock
        start_date: Start date for price data

    Returns:
        tuple: (stock_code, list of price dictionaries)
    """
    try:
        stock_code = f"{stock.code}{market.stock_suffix}"
        price_df, volume_df = get_stock_data(stock_code, start_date)

        if price_df is not None:
            prices_to_add = []
            for idx, row in price_df.iterrows():
                prices_to_add.append({
                    'stock': stock.code,
                    'date': row['timestamp'].to_pydatetime(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(volume_df.iloc[idx]['volume'])
                })
            return stock.code, prices_to_add

        return stock.code, []

    except Exception as e:
        print(f"Error fetching data for {stock_code}: {str(e)}")
        return stock.code, []


def fetch_and_store_stock_data(stocks, start_date, end_date=None, clear_existing=False, max_workers=5):
    """
    Fetch stock data for all stocks and store in database using parallel processing.

    Args:
        stocks: List of stock objects
        start_date: Start date for data fetch
        end_date: End date for data fetch (optional)
        clear_existing: Whether to clear existing price data before storing
        max_workers: Maximum number of concurrent threads
    """
    if arguments["use_existing_price_data"]:
        print("Using existing price data from database...")
        return

    print("Fetching and storing stock price data in concurrent batches...")

    try:
        create_stock_price_table()
    except peewee.OperationalError:
        print("Table already exists")
        if clear_existing:
            print("Clearing existing data...")
            delete_all_stock_prices()

    # Create a lock for thread-safe database operations
    db_lock = Lock()

    # Create a market lookup dictionary to avoid creating Market objects repeatedly
    market_lookup = {stock.exchange: Market(stock.exchange) for stock in stocks}

    def process_batch(batch):
        """Process a batch of stocks and store their prices"""
        all_prices = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each stock in the batch
            future_to_stock = {
                executor.submit(
                    fetch_prices_for_stock,
                    stock,
                    market_lookup[stock.exchange],
                    start_date
                ): stock for stock in batch
            }

            # Process completed futures
            for future in as_completed(future_to_stock):
                stock_code, prices = future.result()
                if prices:
                    all_prices.extend(prices)

        # Store the batch of prices in the database
        if all_prices:
            with db_lock:
                try:
                    bulk_add_stock_prices(all_prices)
                except Exception as e:
                    print(f"Error storing prices: {str(e)}")

    # Process stocks in batches to manage memory usage
    batch_size = 100
    total_batches = (len(stocks) + batch_size - 1) // batch_size

    with tqdm(total=len(stocks), desc='Fetching data') as pbar:
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            process_batch(batch)
            pbar.update(len(batch))


if __name__ == "__main__":

    start_time = time()

    arguments = define_scanner_args()

    # Define the dates
    reporting_date_start = get_data_start_date(arguments["date"])

    # Lookback da for earnings. For simplification, looking 5 days back
    # This will be enought to catch cases with Friday earnings, gap on Monday, and scanning on Tue even with holidays
    current_date, lookback_date = get_current_and_lookback_date(arguments["date"])

    # Initiate market objects
    active_markets = []
    for market_code in config["markets"]:
        active_markets.append(Market(market_code))

    if arguments["update"]:
        update_stocks(active_markets)
    if arguments["stocks"]:
        print(f'Force checking these stock only: {arguments["stocks"]}')

    if arguments["scan"]:
        check_update_date(active_markets)
        scan_stocks(active_markets)

    print()
    end_time = time()
    minutes_passed = (end_time - start_time) // 60
    print(f"{minutes_passed} minutes passed")
