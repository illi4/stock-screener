# Monitors whether the open positions hit the exit criteria at some point
# Output W: wanted exit price, A: actual exit, D4: result if nothing on day 3 (so exit on 4), similar with D6

# Suppress warnings from urllib and gspread
import warnings
warnings.filterwarnings("ignore")

import libs.gsheetobj as gsheetsobj
from libs.stocktools import get_stock_data, Market
from libs.techanalysis import MA, SAR, td_indicators
import arrow
from datetime import timedelta
from libs.helpers import get_data_start_date, define_args_method_only
from libs.signal import market_bearish

from tqdm import tqdm

from scanner import generate_indicators_daily_weekly
import pandas as pd
from datetime import datetime, timedelta

from libs.read_settings import read_config
config = read_config()

reporting_date_start = get_data_start_date()

def get_first_true_idx(list):
    filtr = lambda x: x == True
    return [i for i, x in enumerate(list) if filtr(x)][0]


def check_market(market):
    market_ohlc_daily, market_volume_daily = get_stock_data(market.related_market_ticker, reporting_date_start)
    is_market_bearish, _ = market_bearish(market_ohlc_daily, market_volume_daily, output=True,
                                          verbose_market_name=market.related_market_ticker)
    if is_market_bearish:
        print("Overall market sentiment is bearish, exit all the open positions")
        exit(0)


def check_positions(method_name):
    alerted_positions = set()

    sheet_name = config["logging"]["gsheet_name"]
    tab_name = config["logging"]["gsheet_tab_name"]

    ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)

    for index, row in tqdm(ws.iterrows(), total=len(ws), desc='Processing', unit='stock(s)'):
        if (
            row["Outcome"] == ""
        ):  # exclude the ones where we have results already, check if price falls below MA10

            ma10, mergedDf = None, None

            stock_code = row["Stock"]
            entry_date_value = row["Entry date"]
            # For each stock, have to initiate a method with market params
            market = Market(row["Market"])

            try:
                entry_date = arrow.get(entry_date_value, "DD/MM/YYYY").datetime.date()
            except arrow.parser.ParserMatchError:
                print("Skipping blank entry date lines")
                continue  # continue with the next iteration in the for cycle

            ohlc_daily, volume_daily = get_stock_data(
                f"{stock_code}{market.stock_suffix}", reporting_date_start
            )

            # MRI method
            # Was not checked for correctness of execution after the last update, may not work
            if method_name == 'mri':
                ma10 = MA(ohlc_daily, 10)
                mergedDf = ohlc_daily.merge(ma10, left_index=True, right_index=True)
                mergedDf.dropna(inplace=True, how="any")

                mergedDf["close_below_ma"] = mergedDf["close"].lt(
                    mergedDf["ma10"]*0.9975
                )  # LT is lower than; close below MA10 minus 0.25%

                mergedDf["timestamp"] = mergedDf["timestamp"].dt.date
                mergedDf = mergedDf[
                    mergedDf["timestamp"] >= entry_date
                ]  # only look from the entry date

                if len(mergedDf) == 0:
                    continue  # skip to the next element if there is nothing yet

                # Also find the first date where price was lower than the entry date low
                entry_low = mergedDf["low"].values[0]
                mergedDf["close_below_entry_low"] = mergedDf["low"].lt(
                    entry_low
                )  # LT is lower than

                alert = True in mergedDf["close_below_ma"].values
                if alert:
                    hit_idx = get_first_true_idx(mergedDf["close_below_ma"].values)
                    hit_date = mergedDf["timestamp"].values[hit_idx]

                    if True in mergedDf["close_below_entry_low"].values:
                        hit_lower_than_low_idx = get_first_true_idx(mergedDf["close_below_entry_low"].values)
                        lower_than_low_date = mergedDf["timestamp"].values[hit_lower_than_low_idx]
                    else:
                        lower_than_low_date = "-"

                    exit_date = hit_date + timedelta(days=1)
                    wanted_price = mergedDf["close"].values[hit_idx]
                    ed_low = mergedDf["low"].values[0]
                    ed_low_shifted = ed_low * 0.995  # minus 0.5% roughly
                    ed_open = mergedDf["open"].values[0]
                    entry_day_low_result = (ed_low_shifted - ed_open)/ed_open

                    try:
                        opened_price = mergedDf["open"].values[hit_idx + 1]
                        # results for 4th day open and 6th day open
                        # for 'not moving for 3d / 5d' (first index is 0)
                        alerted_positions.add(
                            f"{stock_code} ({market.market_code}) [{entry_date} -> {exit_date}] "
                            f"wanted price: {round(wanted_price, 3)} | actual price: {round(opened_price, 3)}"
                        )
                        print(
                            f"{stock_code} ({market.market_code}) [{entry_date}]: alert"
                        )
                    except IndexError:
                        alerted_positions.add(
                            f"{stock_code} ({market.market_code}) [{entry_date} -> {exit_date}]"
                        )
                        print(f"{stock_code} ({market.market_code}): alert (market pre-open)")

                else:
                    print(
                        f"{stock_code} ({market.market_code}) [{entry_date}]: on track"
                    )

            elif method_name == 'anx':
                '''
                (
                    ohlc_with_indicators_daily,
                    ohlc_with_indicators_weekly,
                ) = generate_indicators_daily_weekly(ohlc_daily)

                # Get SAR and check for flips # that doesn't seem to work well
                # Maybe revisit https://www.tradingview.com/script/OkACQQgL-Lucid-SAR/
                ohlc_with_indicators_weekly = SAR(ohlc_with_indicators_weekly)
                ohlc_with_indicators_weekly["start_of_week"] = ohlc_with_indicators_weekly["start_of_week"].dt.date
                ohlc_with_indicators_weekly = ohlc_with_indicators_weekly[
                    ohlc_with_indicators_weekly["start_of_week"] >= entry_date
                ]  # only look from the entry date

                # Just a simplified approach
                if -1 in ohlc_with_indicators_weekly["trend"].values:
                    alerted_positions.add(
                        f"{stock_code} ({exchange}): possible SAR flip"
                    )
                '''
                
                # Check the bearish cross
                ma7 = MA(ohlc_daily, 7)
                ma30 = MA(ohlc_daily, 30)
                mergedDf = ohlc_daily.merge(ma7, left_index=True, right_index=True)
                mergedDf.dropna(inplace=True, how="any")
                mergedDf = mergedDf.merge(ma30, left_index=True, right_index=True)
                mergedDf.dropna(inplace=True, how="any")

                mergedDf["timestamp"] = mergedDf["timestamp"].dt.date
                mergedDf = mergedDf[
                    mergedDf["timestamp"] >= entry_date
                ]  # only look from the entry date

                # Define the crossover condition
                condition = (mergedDf['ma7'].shift(1) > mergedDf['ma30'].shift(1)) & (
                            mergedDf['ma7'] < mergedDf['ma30'])

                # Get the rows where the condition is True
                crossovers = mergedDf[condition]

                # Check if there is any crossover
                alert = not crossovers.empty

                if alert:
                    cross_date = mergedDf.loc[condition, 'timestamp'].iloc[0]
                    days_diff = (datetime.now().date() - cross_date).days
                    
                    # Add alert only if the bearish cross date is within 5 days from today
                    if days_diff <= 5:
                        alerted_positions.add(
                            f"(!) {stock_code} ({market.market_code}): recent bearish cross on {cross_date} ({days_diff} days ago)"
                        )
                    else:
                        alerted_positions.add(
                            f"{stock_code} ({market.market_code}): historical bearish cross on {cross_date} ({days_diff} days ago)"
                        )

    return alerted_positions


def check_green_star():
    """
    Check for green star pattern with specific TD setup rules:
    - Must be in a sequence of green TD setup (1-9)
    - Current close must be higher than:
        a) close of candle with TD setup = 1
        b) close of previous candle
    - All three candles must be part of the same green sequence
    - Pattern can only trigger once per green sequence (1-9)
    """
    alerted_positions = set()

    sheet_name = config["logging"]["gsheet_name"]
    tab_name = config["logging"]["watchlist_green_star_tab_name"]

    try:
        ws = gsheetsobj.sheet_to_df(sheet_name, tab_name)
    except Exception as e:
        print(f"Error reading sheet: {e}")
        return alerted_positions

    print(f"Processing {len(ws)} stocks from {tab_name}...")

    for index, row in tqdm(ws.iterrows(), total=len(ws), desc='Processing', unit='stock(s)'):
        stock_code = row["Stock"]
        market = Market(row["Market"])

        try:
            ohlc_daily, volume_daily = get_stock_data(
                f"{stock_code}{market.stock_suffix}", reporting_date_start
            )

            if ohlc_daily is None or len(ohlc_daily) < 10:  # Need enough data for TD setup
                print(f"Insufficient data for {stock_code}")
                continue

            # Calculate TD indicators
            td_values = td_indicators(ohlc_daily)

            # Get latest values
            current_td_direction = td_values['td_direction'].iloc[-1]
            current_td_setup = td_values['td_setup'].iloc[-1]
            current_close = ohlc_daily['close'].iloc[-1]

            # Only proceed if we're in a green TD sequence
            if current_td_direction != 'green' or current_td_setup == 0:
                continue

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
                continue

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
                continue

            # Check current candle for pattern
            previous_close = ohlc_daily['close'].iloc[-2]

            if (current_close > td1_close and
                    current_close > previous_close and
                    td_values['td_direction'].iloc[-2] == 'green'):  # Verify previous candle was also green

                alert_msg = (f"{stock_code} ({market.market_code}) | "
                             f"Current close: ${current_close:.2f} | "
                             f"Previous close: ${previous_close:.2f} | "
                             f"TD1 close: ${td1_close:.2f} | "
                             f"Current TD setup: {current_td_setup}")

                alerted_positions.add(alert_msg)
                print(f"-> Pattern found: {stock_code} ({market.market_code})")
                print(f"   Current TD setup: {current_td_setup}")
                print(f"   Current close: ${current_close:.2f}")
                print(f"   Previous close: ${previous_close:.2f}")
                print(f"   TD1 close: ${td1_close:.2f}")

        except Exception as e:
            print(f"Error processing {stock_code}: {e}")
            continue

    return alerted_positions

if __name__ == "__main__":

    arguments = define_args_method_only()

    # Initiate market objects
    active_markets = []
    for market_code in config["markets"]:
        active_markets.append(Market(market_code))
        
    # print("Checking the markets...")
    # for market in active_markets:
    #     check_market(market)

    print("Checking positions...")
    alerted_positions = check_positions(method_name=arguments["method"])

    # Not used anymore - Earnings drop followed by green star check is implemented in scanner itself
    '''
    if arguments["method"] == "green_star":
        alerted_positions = check_green_star()
    else:
        alerted_positions = check_positions(method_name=arguments["method"])
    '''

    print()
    if len(alerted_positions) == 0:
        print("No alerts")
    else:
        print(f"Rules triggered for {len(alerted_positions)} stock(s):")
        for position in sorted(alerted_positions):
            print(f"- {position}")
