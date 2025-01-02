from libs.techanalysis import MA, StochRSI, coppock_curve, lucid_sar
from libs.helpers import format_bool
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


from libs.read_settings import read_config
config = read_config()


def bullish_breakout(
    ohlc_with_indicators_daily,
    volume_daily,
    ohlc_with_indicators_weekly,
    consider_volume_spike=True,
    output=True,
    stock_name="",
):
    # 3MA variant of a bullish breakout system on volume
    result, numerical_score = bullish_mri_based(
        ohlc_with_indicators_daily,
        volume_daily,
        ohlc_with_indicators_weekly,
        consider_volume_spike,
        output,
        stock_name,
    )
    return result, numerical_score


def slow_ma_inavailable(ma30):
    ma30_nan = np.isnan(ma30["ma30"].iloc[-1])
    return ma30_nan


def ma_consensio(slow_ma_nan, ma_values, number_of_ma):
    """
    :param slow_ma_nan: is the slowest MA none
    :param ma_values: dictionary of MA values (dataframes)
    :param number_of_ma: number of MAs considered (2 or 3)
    :return: bool
    """
    if not slow_ma_nan:
        if number_of_ma == 3:
            is_ma_consensio = (
                ma_values["ma10"]["ma10"].iloc[-1]
                > ma_values["ma20"]["ma20"].iloc[-1]
                > ma_values["ma30"]["ma30"].iloc[-1]
            )
        elif number_of_ma == 2:
            is_ma_consensio = (
                ma_values["ma10"]["ma10"].iloc[-1] > ma_values["ma30"]["ma30"].iloc[-1]
            )
    else:
        is_ma_consensio = False
        print("-- note: MA30 is NaN, the stock is too new")

    return is_ma_consensio


def weekly_close_above_ma(ma_weekly_values, weekly_closes):
    ma30_weekly_nan = np.isnan(ma_weekly_values["ma30"]["ma30"].iloc[-1])
    if not ma30_weekly_nan:
        weekly_conditions = []
        for pit in [1, 2]:
            for ma_checked in [
                ma_weekly_values["ma10"]["ma10"],
                ma_weekly_values["ma20"]["ma20"],
                ma_weekly_values["ma30"]["ma30"],
            ]:
                condition = weekly_closes["close"].iloc[-pit] > ma_checked.iloc[-pit]
                weekly_conditions.append(condition)
        ma_weekly_close_condition = not (False in weekly_conditions)
    else:
        ma_weekly_close_condition = True
        print("-- note: MA30 weekly is NaN, considering weekly close rule as true")

    return ma_weekly_close_condition


def volume_spike(volume_daily):
    volume_ma_20 = MA(volume_daily, 20, colname="volume")
    mergedDf = volume_daily.merge(volume_ma_20, left_index=True, right_index=True)
    mergedDf.dropna(inplace=True, how="any")
    mergedDf["volume_above_average"] = mergedDf["volume"].ge(
        mergedDf["ma20"]*config["filters"]["volume_to_average"]
    )  # GE is greater or equal, than averaged 20d volume x coefficient from settings
    try:
        volume_condition = bool(mergedDf["volume_above_average"].iloc[-1])
    except IndexError:
        print("Issue indexing volume")
        volume_condition = False
    return volume_condition


def ma_increasing(ma_values, number_of_ma):
    if number_of_ma == 3:
        ma_rising = (
            (ma_values["ma10"]["ma10"].iloc[-1] >= ma_values["ma10"]["ma10"].iloc[-3])
            and (
                ma_values["ma20"]["ma20"].iloc[-1] >= ma_values["ma20"]["ma20"].iloc[-3]
            )
            and (
                ma_values["ma30"]["ma30"].iloc[-1] >= ma_values["ma30"]["ma30"].iloc[-3]
            )
        )
    elif number_of_ma == 2:
        ma_rising = (
            ma_values["ma10"]["ma10"].iloc[-1] >= ma_values["ma10"]["ma10"].iloc[-3]
        ) and (ma_values["ma30"]["ma30"].iloc[-1] >= ma_values["ma30"]["ma30"].iloc[-3])
    return ma_rising


def weekly_not_overextended(ohlc_with_indicators_weekly):
    not_overextended = (
        ohlc_with_indicators_weekly["close"].iloc[-1]
        < (1 + config["filters"]["overextended_threshold_percent"] / 100)
        * ohlc_with_indicators_weekly["close"].iloc[-4]
    )
    return not_overextended


def last_is_green(ohlc_with_indicators_daily):
    last_candle_is_green = (
        ohlc_with_indicators_daily["close"].iloc[-1]
        > ohlc_with_indicators_daily["open"].iloc[-1]
    )
    return last_candle_is_green


def recent_close_above_last(ohlc_with_indicators_daily):
    ohlc_with_indicators_daily["candle_body_upper"] = ohlc_with_indicators_daily[
        ["open", "close"]
    ].max(axis=1)
    close_most_recent = float(ohlc_with_indicators_daily["close"].iloc[-1])
    ohlc_with_indicators_daily["lower_than_recent"] = ohlc_with_indicators_daily[
        "candle_body_upper"
    ].lt(
        close_most_recent
    )  # LT is lower than

    # Do not include the most recent itself in the calculation. Take N previous before that.
    candle_idx = config["filters"]["higher_than_n_last_candles"] + 1

    previous_n_lower_than_recent = ohlc_with_indicators_daily["lower_than_recent"][
        -candle_idx:-1
    ].tolist()
    upper_condition = not (False in previous_n_lower_than_recent)
    return upper_condition


def stoch_rsi_in_range(ohlc_with_indicators_daily):
    stoch_rsi_k,  stoch_rsi_d = StochRSI(ohlc_with_indicators_daily)

    stock_rsi_max = max(stoch_rsi_k.iloc[-1], stoch_rsi_d.iloc[-1])
    stoch_rsi_in_range_condition = stock_rsi_max < 0.9  # less than 90% per tests

    return stoch_rsi_in_range_condition


def broad_range(ohlc_with_indicators_weekly):
    last_n_weeks = ohlc_with_indicators_weekly.tail(config["filters"]["range_over_weeks"])

    highest_high = last_n_weeks["high"].max()
    lowest_low = last_n_weeks["low"].min()
    hg_condition = 100*(highest_high/lowest_low - 1) >= config["filters"]["range_percentage"]

    return hg_condition


def bullish_sars(ohlc_with_indicators_weekly):
    return ohlc_with_indicators_weekly['trend'].iloc[-1] == 1


def price_above_ma(ohlc_with_indicators_daily, ma_values, ma_length):
    condition = (
        ohlc_with_indicators_daily["close"].iloc[-1]
        > ma_values[f"ma{ma_length}"].iloc[-1]
    )
    return condition

def coppock_is_positive(ohlc_with_indicators_daily, ohlc_with_indicators_weekly):
    coppock_daily = coppock_curve(ohlc_with_indicators_daily).iloc[-1].values[0]
    coppock_weekly = coppock_curve(ohlc_with_indicators_weekly).iloc[-1].values[0]
    condition = (coppock_daily > 0) and (coppock_weekly > 0)
    return condition

def recent_bullish_cross(ma_a, ma_b, a_length, b_length):
    return (
                ma_a[f"ma{a_length}"].iloc[-1] > ma_b[f"ma{b_length}"].iloc[-1]
                    and
                ma_a[f"ma{a_length}"].iloc[-2] < ma_b[f"ma{b_length}"].iloc[-2]
    )

def recent_bearish_cross(ma_a, ma_b, a_length, b_length):
    return (
                ma_a[f"ma{a_length}"].iloc[-1] < ma_b[f"ma{b_length}"].iloc[-1]
                    and
                ma_a[f"ma{a_length}"].iloc[-2] > ma_b[f"ma{b_length}"].iloc[-2]
    )


def price_crossed_ma(ohlc_daily, ma_values_faster, ma_length_faster, ma_values_slower, ma_length_slower):
    """
    Check if price crossed above MA and closed above it on the most recent candle,
    ensuring only the current candle touches the MA, not the previous one.

    Args:
        ohlc_daily: DataFrame with OHLC data
        ma_values_faster: MA values DataFrame
        ma_length_faster: Length of MA to check
        ma_values_slower: MA values DataFrame
        ma_length_slower: Length of MA to check

    Returns:
        bool: True if price crossed and closed above MA under the specified conditions
    """
    if len(ohlc_daily) < 2:
        return False

    # Check that the previous candle was entirely above the MA
    prev_candle_touches = (
        ohlc_daily["low"].iloc[-2] <= ma_values_slower[f"ma{ma_length_slower}"].iloc[-2]
    )

    # Current candle should touch and close above MA
    curr_touches_ma = (
        ohlc_daily["low"].iloc[-1] <= ma_values_slower[f"ma{ma_length_slower}"].iloc[-1]
    )
    curr_close_above = (
        ohlc_daily["close"].iloc[-1] > ma_values_slower[f"ma{ma_length_slower}"].iloc[-1]
    )

    # It should be a green candle
    green_candle = ohlc_daily["close"].iloc[-1] > ohlc_daily["open"].iloc[-1]

    # Fast MA should be above slow MA for both candles
    ma_above_previously = (
        ma_values_faster[f"ma{ma_length_faster}"].iloc[-2] >
        ma_values_slower[f"ma{ma_length_slower}"].iloc[-2]
    )
    ma_above_now = (
        ma_values_faster[f"ma{ma_length_faster}"].iloc[-1] >
        ma_values_slower[f"ma{ma_length_slower}"].iloc[-1]
    )


    # Return true only if current candle touches and closes above MA,
    # previous candle doesn't touch MA, and other conditions are met
    return (
        #not prev_candle_touches and  # Previous candle doesn't touch MA  # dropped this rule
        curr_touches_ma and          # Current candle touches MA
        curr_close_above and         # Current candle closes above MA
        #green_candle and            # Current candle is green  # dropped this rule
        ma_above_previously and     # Fast MA above slow MA on previous candle
        ma_above_now               # Fast MA above slow MA on current candle
    )


def price_gapped_down(ohlc_with_indicators_daily, gap_threshold):
    """
    Check if the latest day's open price gapped down from previous day's lowest of open/close by more than threshold percentage.

    Args:
    ohlc_with_indicators_daily (pd.DataFrame): DataFrame containing OHLC data
    gap_threshold (float): Minimum gap percentage required (in decimal form)

    Returns:
    bool: True if price gapped down by more than threshold, False otherwise
    """
    if len(ohlc_with_indicators_daily) < 2:
        return False

    # Get the lowest of previous day's open and close
    previous_lowest = min(ohlc_with_indicators_daily["open"].iloc[-2], ohlc_with_indicators_daily["close"].iloc[-2])

    # Get current open
    current_lowest = min(ohlc_with_indicators_daily["open"].iloc[-1], ohlc_with_indicators_daily["close"].iloc[-1])
    gap_percent = (previous_lowest - current_lowest) / previous_lowest

    gap_condition = gap_percent > gap_threshold

    if gap_condition:
        print(
            f"- Gap down detected: {gap_percent:.1%} | Previous lowest (open/close): ${previous_lowest:.2f} | Current open: ${current_lowest:.2f}")

    return gap_condition


def earnings_gap_down(
        ohlc_with_indicators_daily,
        volume_daily,
        ohlc_with_indicators_weekly,
        output=True,
        stock_name="",
):
    """
    Check for earnings gap down signal based on configured threshold

    Args:
    ohlc_with_indicators_daily (pd.DataFrame): Daily OHLC data with indicators
    volume_daily (pd.DataFrame): Daily volume data
    ohlc_with_indicators_weekly (pd.DataFrame): Weekly OHLC data with indicators
    output (bool): Whether to print output messages
    stock_name (str): Name of the stock for output messages

    Returns:
    tuple: (bool, int) - Signal confirmation and numerical score
    """
    from libs.read_settings import read_config
    config = read_config()

    # Get gap threshold from config
    gap_threshold = config["filters"].get("earnings_gap_threshold", None)

    # Check for gap down
    gap_down_condition = price_gapped_down(ohlc_with_indicators_daily, gap_threshold)

    if output:
        print(
            f"- {stock_name} | "
            f"Gap down condition: [{format_bool(gap_down_condition)}]"
        )

    confirmation = [gap_down_condition]

    # Score is either 5 (confirmed) or 0 (not confirmed)
    result = False not in confirmation
    numerical_score = 5 if result else 0

    return result, numerical_score


def bullish_mri_based(
    ohlc_with_indicators_daily,
    volume_daily,
    ohlc_with_indicators_weekly,
    consider_volume_spike=True,
    output=True,
    stock_name="",
):
    """
    :param ohlc_with_indicators_daily: daily OHLC with indicators (pandas df)
    :param volume_daily: volume values (pandas df)
    :param ohlc_with_indicators_weekly: weekly OHLC with indicators (pandas df)
    :param consider_volume_spike: is the volume spike condition considered
    :param output: should the output be printed
    :param stock_name: name of a stock
    :return:
    """
    ma_num_considered = 3  # number of MAs to use

    daily_condition_close_higher = (  # closes higher
        ohlc_with_indicators_daily["close"].iloc[-1]
        > ohlc_with_indicators_daily["close"].iloc[-2]
    )
    daily_condition_td = (  # bullish TD count
        ohlc_with_indicators_daily["td_direction"].iloc[-1] == "green"
    )
    weekly_condition_td = (  # bullish TD count
        ohlc_with_indicators_weekly["td_direction"].iloc[-1] == "green"
    )

    # MA daily
    ma10 = MA(ohlc_with_indicators_daily, 10)
    ma20 = MA(ohlc_with_indicators_daily, 20)
    ma30 = MA(ohlc_with_indicators_daily, 30)
    ma_daily_values = dict(ma10=ma10, ma20=ma20, ma30=ma30,)

    # MA weekly
    ma10_weekly = MA(ohlc_with_indicators_weekly, 10)
    ma20_weekly = MA(ohlc_with_indicators_weekly, 20)
    ma30_weekly = MA(ohlc_with_indicators_weekly, 30)
    ma_weekly_values = dict(ma10=ma10_weekly, ma20=ma20_weekly, ma30=ma30_weekly,)

    # MA30 may be None for too new stocks
    slow_ma_nan = slow_ma_inavailable(ma30)

    # Factor: ma consensio
    is_ma_consensio = ma_consensio(slow_ma_nan, ma_daily_values, ma_num_considered)

    # Factor: weekly close is higher than MAs for the last 2 closes
    ma_weekly_close_condition = weekly_close_above_ma(
        ma_weekly_values, ohlc_with_indicators_weekly
    )

    # Factor: Volume MA and volume spike over the considered day
    if consider_volume_spike:
        volume_condition = volume_spike(volume_daily)
    else:
        volume_condition = True

    # Factor: All MAs are rising
    ma_rising = ma_increasing(ma_daily_values, ma_num_considered)

    # Factor: Close for the last week is not more than X% from the 4 weeks ago
    not_overextended = weekly_not_overextended(ohlc_with_indicators_weekly)

    # Factor: Last candle should actually be green (close above open)
    last_candle_is_green = last_is_green(ohlc_with_indicators_daily)

    # Factor: Most recent close should be above the bodies of 5 candles prior
    upper_condition = recent_close_above_last(ohlc_with_indicators_daily)

    # Factor: stochastic RSI below 90
    stoch_rsi_condition = stoch_rsi_in_range(ohlc_with_indicators_daily)

    # Factor: Must be high growth and not just barely moving
    broad_range_condition = broad_range(ohlc_with_indicators_weekly)

    if output:
        print(
            f"- {stock_name} MRI: D [{format_bool(daily_condition_td)}] / W [{format_bool(weekly_condition_td)}] | "
            f"Consensio: [{format_bool(is_ma_consensio)}] | MA rising: [{format_bool(ma_rising)}] | "
            f"Not overextended: [{format_bool(not_overextended)}] \n"
            f"- {stock_name} Higher close: [{format_bool(daily_condition_close_higher)}] | "
            f"Volume condition: [{format_bool(volume_condition)}] | Upper condition: [{format_bool(upper_condition)}] | "
            f"Last candle is green: [{format_bool(last_candle_is_green)}] | "
            f"Broad range condition:  [{format_bool(broad_range_condition)}] | "
            f"StochRSI not overextended: [{format_bool(stoch_rsi_condition)}] | "
            f"Weekly/MA close: [{format_bool(ma_weekly_close_condition)}]"
        )

    confirmation = [
        daily_condition_td,
        weekly_condition_td,
        is_ma_consensio,
        ma_rising,
        not_overextended,
        daily_condition_close_higher,
        volume_condition,
        upper_condition,
        last_candle_is_green,
        ma_weekly_close_condition,
        broad_range_condition,
        stoch_rsi_condition
    ]
    numerical_score = round(
        5 * sum(confirmation) / len(confirmation), 0
    )  # score X (of 5)
    result = False not in confirmation

    return result, numerical_score


def is_ma_rising(ma_values, ma_length, lookback_period=5, spread=2):
    """
    Check if moving average is rising by comparing points spread around each value

    Args:
        ma_values: DataFrame with MA values
        ma_length: Length of MA (for column name)
        lookback_period: Number of points to check
        spread: Number of periods to look back for each comparison

    Returns:
        bool: True if MA is rising over the period
    """
    if len(ma_values) < lookback_period + spread:
        return False

    ma_series = ma_values[f'ma{ma_length}'].tail(lookback_period + spread)
    # Compare each point with a point 'spread' periods before
    rising_checks = []
    for i in range(spread, len(ma_series)):
        rising_checks.append(ma_series.iloc[i] > ma_series.iloc[i - spread])

    # Calculate what percentage of checks were true
    rising_percentage = sum(rising_checks) / len(rising_checks)
    # Return True if at least 80% of checks showed rising values
    return rising_percentage >= 0.8


def is_bullish_sar(sar_values):
    # Check whether SAR indicates uptrend
    return (sar_values["uptrend"].iloc[-1])

def is_bearish_sar(sar_values):
    # Check whether SAR indicates uptrend
    return (not sar_values["uptrend"].iloc[-1])

def check_recent_green_candle(ohlc_daily, lookback=3):
    """
    Check if there's at least one green candle in the most recent N candles.

    Args:
    ohlc_daily (pd.DataFrame): DataFrame with OHLC data
    lookback (int): Number of recent candles to check

    Returns:
    bool: True if there's at least one green candle in the period
    """
    recent_candles = ohlc_daily.tail(lookback)
    green_candles = (recent_candles['close'] > recent_candles['open']).any()
    return green_candles


def check_max_drawdown(ohlc_daily, lookback=14, max_drawdown_percent=0.15):
    """
    Check if the stock hasn't dropped more than specified percentage from recent high.

    Args:
    ohlc_daily (pd.DataFrame): DataFrame with OHLC data
    lookback (int): Number of recent candles to check
    max_drawdown_percent (float): Maximum allowed drawdown as decimal

    Returns:
    bool: True if drawdown is within acceptable range
    """
    recent_data = ohlc_daily.tail(lookback)
    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()

    drawdown = (recent_high - recent_low) / recent_high
    return drawdown <= max_drawdown_percent


def check_wick_conditions(ohlc_daily, lookback=5, max_wick_bodies=2):
    """
    Check if recent candles don't have significant upper wicks.
    Upper wick should not be larger than max_wick_bodies times the candle's range.

    Args:
    ohlc_daily (pd.DataFrame): DataFrame with OHLC data
    lookback (int): Number of recent candles to check
    max_wick_bodies (float): Maximum allowed wick size as multiple of candle range

    Returns:
    bool: True if wick conditions are met
    """
    recent_candles = ohlc_daily.tail(lookback)

    for _, candle in recent_candles.iterrows():
        body_high = max(candle['open'], candle['close'])
        candle_range = abs(candle['high'] - candle['low'])
        upper_wick = candle['high'] - body_high

        # Skip check if candle range is too small to avoid division by zero
        if candle_range < 0.0001:
            continue

        # Compare upper wick to the candle's range
        if upper_wick > (candle_range * max_wick_bodies):
            return False

    return True


def bullish_anx_based(
        ohlc_with_indicators_daily,
        volume_daily,
        ohlc_with_indicators_weekly,
        output=True,
        stock_name="",
):
    # Read config for strategy settings
    config = read_config()
    trigger_type = config["strategy"]["anx"]["trigger_type"]

    # Lucid SAR calculations
    sar_values = lucid_sar(ohlc_with_indicators_weekly)   # the lucid sar itself works well

    # Check for the uptrend Lucid SAR conditions
    bullish_sar_condition = is_bullish_sar(sar_values)

    # MA calculations
    ma3 = MA(ohlc_with_indicators_daily, length=3, ma_type='exponential')
    ma12 = MA(ohlc_with_indicators_daily, length=12, ma_type='exponential')
    ma50 = MA(ohlc_with_indicators_daily, length=50, ma_type='exponential')
    ma200 = MA(ohlc_with_indicators_daily, length=200, ma_type='exponential')

    # Check trigger conditions based on config
    ma_cross_condition = False
    price_cross_condition = False

    if trigger_type in ["ma_cross", "both"]:
        ma_cross_condition = recent_bullish_cross(ma3, ma12, 3, 12)

    if trigger_type in ["price_cross", "both"]:
        price_cross_condition = price_crossed_ma(ohlc_with_indicators_daily, ma3, 3, ma12, 12)

    # Combined trigger condition based on strategy setting
    if trigger_type == "both":
        trigger_condition = ma_cross_condition or price_cross_condition
    else:
        trigger_condition = ma_cross_condition if trigger_type == "ma_cross" else price_cross_condition

    # Create note about which condition triggered
    trigger_note = ""
    if ma_cross_condition and trigger_type != "price_cross":
        trigger_note = "[MA3/MA12 bullish cross]"
    elif price_cross_condition and trigger_type != "ma_cross":
        trigger_note = "[Price crossed above MA12]"

    # Other existing conditions
    price_above_ma_condition = price_above_ma(ohlc_with_indicators_daily, ma200, 200)
    not_overextended = weekly_not_overextended(ohlc_with_indicators_weekly)

    # Add MA50 rising condition with spread comparison
    ma50_rising = is_ma_rising(ma50, 50, lookback_period=5, spread=2)

    # Add new conditions when looking at price cross rather than MA cross
    if trigger_type in ["price_cross", "both"]:
        recent_green_condition = check_recent_green_candle(ohlc_with_indicators_daily)
        drawdown_condition = check_max_drawdown(ohlc_with_indicators_daily)
        wick_condition = check_wick_conditions(ohlc_with_indicators_daily)
    else:
        recent_green_condition = drawdown_condition = wick_condition = True

    if output:
        # Get actual MA50 values for detailed output
        ma50_values = ma50['ma50'].tail(5)
        ma50_current = ma50_values.iloc[-1]
        ma50_prev = ma50_values.iloc[-3]  # Looking 2 periods back
        ma50_change = (ma50_current - ma50_prev) / ma50_prev * 100

        print(
            f"- {stock_name} | "
            f"Strategy type: {trigger_type} | "
            f"Price above MA200: [{format_bool(price_above_ma_condition)}] | "
            f"Price trigger: [{format_bool(trigger_condition)}] {trigger_note} | "
            f"Bullish weekly SAR: [{format_bool(bullish_sar_condition)}] | "
            f"MA50 rising: [{format_bool(ma50_rising)}] ({ma50_change:+.2f}%) | "
            f"not overextended: [{format_bool(not_overextended)}] | "
            f"Recent green OK: [{format_bool(recent_green_condition)}] | "
            f"Drawdown OK: [{format_bool(drawdown_condition)}] | "
            f"Wick OK: [{format_bool(wick_condition)}]"
        )

    confirmation = [
        price_above_ma_condition,
        trigger_condition,
        bullish_sar_condition,
        ma50_rising,
        not_overextended,
        recent_green_condition,
        drawdown_condition,
        wick_condition
    ]

    result = False not in confirmation
    numerical_score = 5  # not used but keep for the output structure

    return result, numerical_score, trigger_note

def bearish_anx_based(
        ohlc_with_indicators_daily,
        volume_daily,
        ohlc_with_indicators_weekly,
        output=True,
        stock_name="",
):
    # Read config for strategy settings
    config = read_config()
    trigger_type = config["strategy"]["anx"]["trigger_type"]

    # Lucid SAR calculations
    sar_values = lucid_sar(ohlc_with_indicators_weekly)   # the lucid sar itself works well

    # Check for the uptrend Lucid SAR conditions
    # TODO: add check of bearish cross happening on the wave change, not just inside the wave
    bearish_sar_condition = is_bearish_sar(sar_values)

    # MA calculations
    ma3 = MA(ohlc_with_indicators_daily, length=3, ma_type='exponential')
    ma12 = MA(ohlc_with_indicators_daily, length=12, ma_type='exponential')
    ma50 = MA(ohlc_with_indicators_daily, length=50, ma_type='exponential')
    ma200 = MA(ohlc_with_indicators_daily, length=200, ma_type='exponential')

    # Check trigger conditions based on config
    ma_cross_condition = False
    price_cross_condition = False

    if trigger_type in ["price_cross", "both"]:
        raise NotImplementedError("Price cross triger not supported for the bearish direction")

    ma_cross_condition = recent_bearish_cross(ma3, ma12, 3, 12)
    trigger_condition = ma_cross_condition
    trigger_note = "[MA3/MA12 bearish cross]"

    if output:
        # Get actual MA50 values for detailed output
        # ma50_values = ma50['ma50'].tail(5)
        # ma50_current = ma50_values.iloc[-1]
        # ma50_prev = ma50_values.iloc[-3]  # Looking 2 periods back
        # ma50_change = (ma50_current - ma50_prev) / ma50_prev * 100

        print(
            f"- {stock_name} | "
            f"Strategy type: {trigger_type} | " 
            f"Price trigger: [{format_bool(trigger_condition)}] {trigger_note} | "
            f"Bearish weekly SAR: [{format_bool(bearish_sar_condition)}] | "
        )

    confirmation = [
        trigger_condition,
        bearish_sar_condition
    ]

    result = False not in confirmation
    numerical_score = 5  # not used but keep for the output structure

    return result, numerical_score, trigger_note

def red_day_on_volume(
    ohlc_with_indicators_daily,
    volume_daily,
    output=False,
    stock_name="",
):
    """
    :param ohlc_with_indicators_daily: daily OHLC with indicators (pandas df)
    :param volume_daily: volume values (pandas df)
    :param ohlc_with_indicators_weekly: weekly OHLC with indicators (pandas df)
    :param consider_volume_spike: is the volume spike condition considered
    :param output: should the output be printed
    :param stock_name: name of a stock
    :return:
    """
    daily_red_close = (
        ohlc_with_indicators_daily["close"].iloc[-1]
        < ohlc_with_indicators_daily["open"].iloc[-1]
    )

    volume_ma_20 = MA(volume_daily, 20, colname="volume")

    mergedDf = volume_daily.merge(volume_ma_20, left_index=True, right_index=True)
    mergedDf.dropna(inplace=True, how="any")
    mergedDf["volume_above_ma"] = mergedDf["volume"].ge(
        mergedDf["ma20"]
    )  # GE is greater or equal, than averaged 20d volume plus 30 percent
    try:
        volume_condition = bool(mergedDf["volume_above_ma"].iloc[-1])
    except IndexError:
        print("Issue indexing volume")
        volume_condition = False

    if output:
        print(
            f"- {stock_name} Red day close: [{daily_red_close}] / volume_condition [{volume_condition}]"
        )

    confirmation = [
        daily_red_close,
        volume_condition
    ]
    numerical_score = round(
        5 * sum(confirmation) / len(confirmation), 0
    )  # score X (of 5)
    result = False not in confirmation

    return result, numerical_score


def market_bearish(
    ohlc_with_indicators_daily,
    volume_daily,
    output=False,
    verbose_market_name=''
):
    """
    :param ohlc_with_indicators_daily: daily OHLC with indicators (pandas df)
    :param volume_daily: volume values (pandas df)
    :param ohlc_with_indicators_weekly: weekly OHLC with indicators (pandas df)
    :param consider_volume_spike: is the volume spike condition considered
    :param output: should the output be printed
    :param stock_name: name of a stock
    :return:
    """
    ma200 = MA(ohlc_with_indicators_daily, 200)
    ma10 = MA(ohlc_with_indicators_daily, 10)

    # Condition: market is below MA200
    market_below_ma_200 = (
        ohlc_with_indicators_daily["close"].iloc[-1]
        < ma200["ma200"].iloc[-1]
    )
    # Condition: MA is decreasing
    ma_10_decreasing = (
            (ma10["ma10"].iloc[-1] < ma10["ma10"].iloc[-2]) and
            (ma10["ma10"].iloc[-1] < ma10["ma10"].iloc[-3]) and
            (ma10["ma10"].iloc[-1] < ma10["ma10"].iloc[-5])
    )

    if output:
        print(
            f"- Market {verbose_market_name} below MA200: [{market_below_ma_200}] | MA10 decreasing: [{ma_10_decreasing}]"
        )

    negative_confirmation = [
        market_below_ma_200,
        ma_10_decreasing
    ]
    numerical_score = round(
        5 * sum(negative_confirmation) / len(negative_confirmation), 0
    )  # score X (of 5)
    result = False not in negative_confirmation

    return result, numerical_score
