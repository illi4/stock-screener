from libs.techanalysis import MA, StochRSI, SAR
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


def recent_bullish_cross(ma_a, ma_b, a_length, b_length):
    return (
                ma_a[f"ma{a_length}"].iloc[-1] > ma_b[f"ma{b_length}"].iloc[-1]
                    and
                ma_a[f"ma{a_length}"].iloc[-2] < ma_b[f"ma{b_length}"].iloc[-2]
    )


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


def bullish_anx_based(
    ohlc_with_indicators_daily,
    volume_daily,
    ohlc_with_indicators_weekly,
    output=True,
    stock_name="",
):
    """
    :param ohlc_with_indicators_daily: daily OHLC with indicators (pandas df)
    :param volume_daily: volume values (pandas df)
    :param ohlc_with_indicators_weekly: weekly OHLC with indicators (pandas df)
    :param output: should the output be printed
    :param stock_name: name of a stock
    :return:
    """

    # Parabolic SARS calculation for checking the trend (1: bullish, -1: bearish)
    # Factor: Weekly SARS wave is bullish
    # The result is inconsistent with ANX indicator on tradingview so not using it, has to be checked manually
    '''
    ohlc_with_indicators_weekly = SAR(ohlc_with_indicators_weekly)
    bullish_sars_condition = bullish_sars(ohlc_with_indicators_weekly)
    '''

    # Factor: price is above MA200
    ma200 = MA(ohlc_with_indicators_daily, 200)
    price_above_ma_condition = price_above_ma(ohlc_with_indicators_daily, ma200, 200)

    # Factor: bullish MA cross (MA7 and MA30)
    ma7 = MA(ohlc_with_indicators_daily, 7)
    ma30 = MA(ohlc_with_indicators_daily, 30)
    recent_bullish_cross_condition = recent_bullish_cross(ma7, ma30, 7, 30)

    # Factor: Close for the last week is not more than X% from the 4 weeks ago
    not_overextended = weekly_not_overextended(ohlc_with_indicators_weekly)

    if output:
        print(
            f"- {stock_name} | "
            f"Price above MA200: [{format_bool(price_above_ma_condition)}] | "
            f"Recent bullish cross: [{format_bool(recent_bullish_cross_condition)}] | not overextended: [{format_bool(not_overextended)}]"
        )

    confirmation = [
        price_above_ma_condition,
        recent_bullish_cross_condition,
        not_overextended
    ]
    numerical_score = round(
        5 * sum(confirmation) / len(confirmation), 0
    )  # score X (of 5)
    result = False not in confirmation

    return result, numerical_score


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
            f"- Market below MA200: [{market_below_ma_200}] | MA10 decreasing: [{ma_10_decreasing}]"
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
