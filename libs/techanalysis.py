import pandas as pd
import numpy as np


def combined_indicators(df):
    """
    Function to return a dataframe of all used indicators calculated from the input dataframe.
    :param df: pandas OHLC dataframe
    :return: pandas dataframe with indicator values
    """

    # To re-do back to dynamic which does not work now
    td_val = td_indicators(df)
    atr_val = ATR(df).drop("timestamp", axis=1)
    crsi_val = CRSI(df).drop("timestamp", axis=1)
    ma10_val = MA(df, 10).drop("timestamp", axis=1)
    ma100 = MA(df, 100).drop("timestamp", axis=1)
    ma20 = MA(df, 20).drop("timestamp", axis=1)
    ma30 = MA(df, 30).drop("timestamp", axis=1)
    rsi14 = RSI(df, 14).drop("timestamp", axis=1)
    rsi2 = RSI(df, 2).drop("timestamp", axis=1)
    rsi3 = RSI(df, 3).drop("timestamp", axis=1)

    resulting_df = pd.concat(
        [td_val, atr_val, crsi_val, ma10_val, ma100, ma20, ma30, rsi14, rsi2, rsi3],
        axis=1,
    )

    return resulting_df


def df_from_series(data_series, timestamp_series, columns):
    """
    Function to generate a pandas dataframe from pandas series
    :param data_series: pandas series with the data
    :param timestamp_series: pandas series with the timestamp
    :param columns: list of columns (e.g. ['rsi'])
    :return: pandas dataframe
    """
    df = pd.DataFrame(data_series)
    df.columns = columns
    # df["timestamp"] = timestamp_series # doesn't work
    return df


def RSI(df, length=14, colname="close"):
    """
    Function to calculate RSI
    :param df: pandas dataframe which has the 'colname' column
    :param length: rsi length (14 is the standard)
    :param colname: column which is used to calculate rsi
    :return: pandas dataframe with columns rsi and timestamp
    """
    delta = df[colname].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=length - 1, adjust=False).mean()
    rDown = down.ewm(com=length - 1, adjust=False).mean().abs()
    rsi = 100 - 100 / (1 + rUp / rDown)

    rsi_df = df_from_series(
        data_series=rsi,
        timestamp_series=df.tail(len(rsi))["timestamp"],
        columns=[f"rsi{length}"],
    )

    return rsi_df


def ADX(df, length=14):
    """
    Function to calculate ADX
    :param df: pandas dataframe which has the 'close', 'high', 'low' column
    :param length: length (14 is the standard)
    :return: pandas dataframe with columns adx and timestamp
    """
    adxind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], n=14)
    adx_val = adxind.adx()

    adx_df = df_from_series(
        data_series=adx_val, timestamp_series=df["timestamp"], columns=["adx"]
    )

    return adx_df


def fisher_distance(df: pd.DataFrame, fisher_length: int = 9, ema_length: int = 50) -> pd.DataFrame:
    """
    Calculate the Fisher Transform with Distance from EMA.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' price column.
    fisher_length (int, optional): The lookback period for the Fisher Transform. Defaults to 9.
    ema_length (int, optional): The period for the EMA calculation. Defaults to 50.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated Fisher values.
    """
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_length, adjust=False).mean()
    df['distFromEMA'] = df['close'] - df['ema']

    high = df['distFromEMA'].rolling(window=fisher_length).max()
    low = df['distFromEMA'].rolling(window=fisher_length).min()

    def round_(val):
        if val > 0.99:
            return 0.999
        elif val < -0.99:
            return -0.999
        else:
            return val

    value = np.zeros(len(df))
    for i in range(1, len(df)):
        if pd.notna(high[i]) and pd.notna(low[i]):
            max_diff = max(high[i] - low[i], 0.001)
            norm_value = 0.66 * ((df['distFromEMA'][i] - low[i]) / max_diff - 0.5) + 0.67 * value[i-1]
            value[i] = round_(norm_value)

    fisher_dist = np.zeros(len(df))
    for i in range(1, len(df)):
        fisher_dist[i] = 0.5 * np.log((1 + value[i]) / max(1 - value[i], 0.001)) + 0.5 * fisher_dist[i-1]

    df['distance'] = fisher_dist
    #df['trigger'] = df['distance'].shift(1)  # this is unnecessary

    return df[['distance']]


def coppock_curve(df: pd.DataFrame, wma_length: int = 10, long_roc_length: int = 14, short_roc_length: int = 11) -> pd.DataFrame:
    """
    Calculate the Coppock Curve.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' price column.
    wma_length (int, optional): The smoothing length for the WMA. Defaults to 10.
    long_roc_length (int, optional): The lookback period for the long ROC. Defaults to 14.
    short_roc_length (int, optional): The lookback period for the short ROC. Defaults to 11.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated Coppock Curve values.
    """
    df = df.copy()

    # Calculate the Rate of Change (ROC) and handle potential anomalies
    df['ROC_long'] = df['close'].pct_change(periods=long_roc_length).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df['ROC_short'] = df['close'].pct_change(periods=short_roc_length).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

    # Calculate the Coppock Curve (sum of the two ROCs)
    df['Coppock'] = df['ROC_long'] + df['ROC_short']

    # Calculate the Weighted Moving Average (WMA)
    def weighted_moving_average(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    df['Coppock_WMA'] = weighted_moving_average(df['Coppock'], wma_length)

    return df[['Coppock_WMA']]  # df[['Close', 'ROC_long', 'ROC_short', 'Coppock', 'Coppock_WMA']]


def lucid_sar(df: pd.DataFrame,
              af_initial: float = 0.02,
              af_increment: float = 0.02,
              af_maximum: float = 0.2) -> pd.DataFrame:
    """
    Calculate the Parabolic SAR (Stop And Reverse) technical indicator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' price columns
    af_initial : float
        Initial acceleration factor (default: 0.02)
    af_increment : float
        Acceleration factor increment (default: 0.02)
    af_maximum : float
        Maximum acceleration factor (default: 0.2)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sar, uptrend, ep, new_trend

    Note: Be cautious of stock splits as they can affect calculations.
    """
    high, low = df['high'].values, df['low'].values
    size = len(high)

    # Initialize arrays
    sar = np.zeros(size)
    uptrend = np.zeros(size, dtype=bool)
    ep = np.zeros(size)
    new_trend = np.zeros(size, dtype=bool)
    af = np.zeros(size)

    # Set initial values
    sar[0] = low[0]
    ep[0] = high[0]
    uptrend[0] = True
    af[0] = af_initial

    for i in range(1, size):
        # Update extreme point and acceleration factor
        ep[i] = max(high[i], ep[i - 1]) if uptrend[i - 1] else min(low[i], ep[i - 1])
        af[i] = (af_initial if new_trend[i - 1] else
                 min(af_maximum, af[i - 1] + af_increment) if ep[i] != ep[i - 1] else af[i - 1])

        # Calculate base SAR
        sar[i] = sar[i - 1] + af[i] * (ep[i] - sar[i - 1])

        # Handle uptrend
        if uptrend[i - 1]:
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if sar[i] > low[i]:
                uptrend[i] = False
                new_trend[i] = True
                sar[i] = max(high[i], ep[i - 1])
                ep[i] = min(low[i], low[i - 1])
            else:
                uptrend[i] = True

        # Handle downtrend
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if sar[i] < high[i]:
                uptrend[i] = True
                new_trend[i] = True
                sar[i] = min(low[i], ep[i - 1])
                ep[i] = max(high[i], high[i - 1])
            else:
                uptrend[i] = False

    return pd.DataFrame({
        'sar': sar,
        'uptrend': uptrend,
        'ep': ep,
        'new_trend': new_trend
    }, index=df.index)


def MA(df, length, colname="close", ma_type="simple"):
    """
    Function to calculate MA (Moving Average)
    :param df: pandas dataframe which has the value column
    :param length: MA length (10/20/30)
    :param colname: column name to use ("close") by default
    :param ma_type: type of moving average ("simple" or "exponential"), default is "simple"
    :return: pandas dataframe with columns MA10/MA20/... and timestamp
    """

    if ma_type.lower() == "simple":
        # Calculate the Simple MA values
        ma_rolling = df[colname].rolling(window=length, min_periods=length).mean()
    elif ma_type.lower() == "exponential":
        # Calculate the Exponential MA values
        ma_rolling = df[colname].ewm(span=length, adjust=False).mean()
    else:
        raise ValueError("Invalid ma_type. Choose 'simple' or 'exponential'.")

    ma_df = df_from_series(
        data_series=ma_rolling,
        timestamp_series=df.tail(len(ma_rolling))["timestamp"],
        columns=[f"ma{length}"],
    )

    return ma_df


def wwma(values, length):
    """
    Functions to calculate WWMA from a dataframe. Used for ATR.
    :param df: pandas series
    :param length: WWMA length
    :return: pandas series
    """
    return values.ewm(alpha=1 / length, adjust=False).mean()


def ATR(df, length=14):
    """
    Function to calculate ATR (Average True Range)
    :param df: pandas dataframe which has the 'close' column
    :param length: ATR length
    :return: pandas dataframe with column atr, atr_to_close, and timestamp
    """

    data = df.copy()  # for calculations

    high = data["high"]
    low = data["low"]
    close = data["close"]
    data["tr0"] = abs(high - low)
    data["tr1"] = abs(high - close.shift())
    data["tr2"] = abs(low - close.shift())

    tr = data[["tr0", "tr1", "tr2"]].max(axis=1)
    atr = wwma(tr, length).to_frame()
    atr.columns = ["atr"]

    atr["close"] = close
    atr["atr_to_close"] = atr["atr"] / atr["close"]
    atr.drop("close", axis=1, inplace=True)

    atr["timestamp"] = df["timestamp"]

    return atr


def ROC(df):
    """
    Function to calculate rate of change
    param df: pandas dataframe which has the 'close' column
    returns: pandas dataframe
    """
    window_len = 100  # default lookback period
    df_closes = pd.DataFrame(df["close"], columns=["close"])
    df_closes["percent_change"] = df_closes["close"].pct_change()
    df_closes["percent_change"].iloc[0] = 0
    df_closes["num_less_than_current"] = (
        df_closes["percent_change"]
        .rolling(window_len + 1, min_periods=1)
        .apply(lambda x: (x[-1] > x[:-1]).sum(), raw=True)
        .astype(int)
    )
    df_closes["roc"] = 100 * df_closes["num_less_than_current"] / window_len
    df_closes.index = df["timestamp"]
    return df_closes["roc"]


def streak(df):
    """
    Function to calculate streaks (up/down in the row)
    param df: pandas dataframe which has the 'close' column
    returns: pandas dataframe
    """
    df_closes = df["close"]

    streak_val = 0
    streak_arr = []
    prev_value = None
    i = 0

    for idx, close in df_closes.items():
        if i > 0:
            if close > prev_value:
                if streak_val <= 0:
                    streak_val = 1
                else:
                    streak_val += 1
            elif close < prev_value:
                if streak_val >= 0:
                    streak_val = -1
                else:
                    streak_val -= 1
            elif close == prev_value:
                streak_val = 0

        streak_arr.append(streak_val)
        prev_value = close
        i += 1

    return_df = pd.DataFrame(streak_arr, columns=["streak"])
    return_df.index = df["timestamp"]

    return return_df


def StochRSI(df, period=14, smoothK=3, smoothD=3):
    """
    Function to calculate Stochastic RSI
    param df: pandas dataframe which has the 'close' column
    returns: K and D values (pandas series)
    """
    series = df['close']

    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period])  #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period])  #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])

    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean()

    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi_K, stochrsi_D


def SAR(barsdata, iaf=0.02, maxaf=0.2):
    # Coincides with PSAR but not with Lucid SAR

    length = len(barsdata)
    high = list(barsdata['high'])
    low = list(barsdata['low'])
    close = list(barsdata['close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]

    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
            barsdata.loc[i, 'PSAR_direction'] = 1
            barsdata.loc[i, 'PSAR_val'] = psar[i]
            barsdata.loc[i, 'af'] = af
        else:
            psarbear[i] = psar[i]
            barsdata.loc[i, 'PSAR_direction'] = -1
            barsdata.loc[i, 'PSAR_val'] = psar[i]
            barsdata.loc[i, 'af'] = af

    return barsdata


def CRSI(df):
    """
    Function to calculate connors rsi
    param df: pandas dataframe which has the 'close' column
    returns: pandas dataframe
    """

    # Calculate ROC
    roc = ROC(df)
    # print('-- roc\n', roc.tail(5)) # debug

    # Get the streak values (UpDown)
    UpDown = streak(df)
    # print('-- updown\n', UpDown.tail(5)) # debug

    # RSI2 of UpDown
    UpDown_rsi = RSI(UpDown.reset_index(), length=2, colname="streak").set_index(
        "timestamp"
    )
    # print('-- UD RSI\n', UpDown_rsi.tail(5)) # debug

    # RSI3 of the close bars
    rsi3 = RSI(df, length=3)
    rsi3.index = df["timestamp"]
    # print('-- rsi3\n', rsi3.tail(5)) # debug

    # Final result
    df_crsi = pd.concat([UpDown_rsi, roc], axis=1)
    df_crsi["crsi"] = (rsi3["rsi3"] + df_crsi["rsi2"] + df_crsi["roc"]) / 3
    # print('FINAL\n', df_crsi.tail(5)) # debug

    return df_crsi.reset_index()[["crsi", "timestamp"]]


def td_indicators(df):
    """
    Function to calculate TD indicator (Tone Vays methodology)
    See http://www.mysmu.edu/faculty/christophert/QF206/Week_05.pdf
    :param bars: pandas dataframe (ohlc)
    :return: pandas dataframe
    """

    # Preparation
    bars = df.copy()
    bearish_flip, bullish_flip = False, False
    setup_up, setup_down = 0, 0
    size = bars["close"].size
    # print "TDLib: Bars df size:", size

    # Calculated fields
    bars.loc[:, "td_setup"] = 0  # additional column: td setup number
    bars.loc[:, "td_direction"] = ""  # td setup direction
    bars.loc[
        :, "move_extreme"
    ] = None  # for stopping when the setup extreme is not calculated rigth

    # Changing the types to preserve memory space
    bars["td_setup"] = pd.to_numeric(bars["td_setup"], errors="coerce")
    bars["move_extreme"] = pd.to_numeric(bars["move_extreme"], errors="coerce")

    # Initial direction and values by default
    direction_up, direction_down = False, False
    move_extreme = None
    countdown_up_flag, countdown_down_flag = False, False
    countdown_up_list = []
    countdown_down_list = []

    # Defining tdst values
    tdst_pre = None
    tdst_level = None

    # Defining correct move_extreme for completed TD1
    move_extreme_pre = None
    move_direction_pre = None

    if bars["close"].iloc[5] > bars["close"].iloc[4]:
        direction_up = True
    elif bars["close"].iloc[5] < bars["close"].iloc[4]:
        direction_down = True

    # Shifting closes for further calculations
    bars["shifted_1"] = bars["close"].shift(1)
    bars["shifted_2"] = bars["close"].shift(2)
    bars["shifted_4"] = bars["close"].shift(4)
    bars["shifted_5"] = bars["close"].shift(5)

    # Comparison operations, resulting in A (above) or B (below)
    bars["bear_flip"] = bars.apply(
        lambda x: True
        if x["shifted_1"] >= x["shifted_5"] and x["close"] <= x["shifted_4"]
        else False,
        axis=1,
    )
    bars["bull_flip"] = bars.apply(
        lambda x: True
        if x["shifted_1"] <= x["shifted_5"] and x["close"] >= x["shifted_4"]
        else False,
        axis=1,
    )
    bars["bear_reset"] = bars.apply(
        lambda x: True if x["close"] > x["shifted_4"] else False, axis=1
    )
    bars["bull_reset"] = bars.apply(
        lambda x: True if x["close"] < x["shifted_4"] else False, axis=1
    )
    # For move_extremes
    bars["shifted_1_low"] = bars["low"].shift(1)
    bars["shifted_2_low"] = bars["low"].shift(2)
    bars["shifted_1_high"] = bars["high"].shift(1)
    bars["shifted_2_high"] = bars["high"].shift(2)
    # Resulting move_extreme
    bars["max_1_2"] = bars[["shifted_1_high", "shifted_2_high"]].max(axis=1)
    bars["min_1_2"] = bars[["shifted_1_low", "shifted_2_low"]].min(axis=1)
    # For countdowns
    bars["if_countdown_down"] = bars.apply(
        lambda x: True if x["close"] <= x["shifted_2_low"] else False, axis=1
    )
    bars["if_countdown_up"] = bars.apply(
        lambda x: True if x["close"] >= x["shifted_2_high"] else False, axis=1
    )

    np_setup = np.zeros(shape=(size, 1), dtype=int)
    np_direction = np.empty(shape=(size, 1), dtype=object)
    np_move_extremes = np.empty(
        shape=(size, 1)
    )  # will have the 1s values per sandy's methodology
    np_move_extremes.fill(np.nan)
    # Countdown
    np_countdown_up = np.zeros(shape=(size, 1), dtype=int)
    np_countdown_down = np.zeros(shape=(size, 1), dtype=int)
    # TDST
    np_tdst = np.empty(shape=(size, 1), dtype=float)
    np_tdst.fill(np.nan)

    ## Looping through
    i = 0
    for bar_row in bars.itertuples():
        if i > 5:  # need 6 candles to start
            ## Price flip
            bearish_flip, bullish_flip = False, False

            if setup_up == 9:
                setup_up = 0  # restart count
            if setup_down == 9:
                setup_down = 0  # restart count

            # Flips - bearish
            if bar_row.bear_flip:
                bearish_flip = True
                direction_down = True
                bullish_flip = False
                tdst_pre = bar_row.high  # high of the move

            # Flips - bullish
            if bar_row.bull_flip:
                bullish_flip = True
                direction_up = True
                bearish_flip = False
                tdst_pre = bar_row.high  # low of the move

            if bearish_flip and direction_up:
                direction_up = False
                setup_down = 1
            if bullish_flip and direction_down:
                direction_down = False
                setup_up = 1

            ## TD Setup (sequential)
            if direction_down and not bearish_flip:
                setup_down += 1  # having it like this fixes the bug with TD1 appearing several times in a row

            if direction_up and not bullish_flip:
                setup_up += 1  # having it like this fixes the bug with TD1 appearing several times in a row

            # Move extreme change calc: 1 -> 2 count it in, not just on flip
            if direction_up and setup_up == 1:
                move_extreme_pre = min(
                    bar_row.open, bar_row.close
                )  # lowest of the candle comparing open and close
                move_direction_pre = "up"
            if direction_down and setup_down == 1:
                move_extreme_pre = max(bar_row.open, bar_row.close)
                move_direction_pre = "down"

            # Move extreme recalculation
            if move_direction_pre == "up" and direction_up and setup_up == 2:
                move_extreme = move_extreme_pre
            if move_direction_pre == "down" and direction_down and setup_down == 2:
                move_extreme = move_extreme_pre

            # Filling the np arrays
            if direction_down:
                np_setup[i] = setup_down
                np_direction[i] = "red"  # down
            if direction_up:
                np_setup[i] = setup_up
                np_direction[i] = "green"  # up

            # Common for any direction
            np_move_extremes[i] = move_extreme

            # Countdowns check
            # Will need to store a list with counters
            # If there is active countdown but we get a 9 in the same direction again - one more countdown is added
            # If a 9 in different direction - the previous direction countdown should stop
            if direction_up and setup_up == 9:
                countdown_up_flag = True
                countdown_down_flag = False  # also in this case delete countdowns
                countdown_down_list = []
                countdown_up_list.append(0)  # reserving place for counter increase
                tdst_level = tdst_pre  # updating the TDST value
            if direction_down and setup_down == 9:
                countdown_down_flag = True
                countdown_up_flag = False
                countdown_up_list = []
                countdown_down_list.append(0)  # reserving place for counter increase
                tdst_level = tdst_pre  # updating the TDST value

            # Writing TDST on every iteration
            np_tdst[i] = tdst_level

            """ TD seq buy: 
            If bar 9 has a close less than or equal to the low of two bars earlier
                then bar 9 becomes 1 countdown
            If not met - then countdown 1 postponed until condition is met and continues until total of 13 closes
            Each should be less or equal to the low 2 bars earlier
            If one of elements in on 13, delete it.
            This is a simplified approach as completion of 13 also requires comparison with 8th setup bar
            """
            if direction_up and countdown_up_flag and bar_row.if_countdown_up:
                countdown_up_list = [x + 1 for x in countdown_up_list]
                countdown_up_list = [x for x in countdown_up_list if x != 13]
            if countdown_up_list != []:
                np_countdown_up[i] = max(
                    countdown_up_list
                )  # this is enough for our purposes
            if direction_down and countdown_down_flag and bar_row.if_countdown_down:
                countdown_down_list = [x + 1 for x in countdown_down_list]
                countdown_down_list = [x for x in countdown_down_list if x != 13]
            if countdown_down_list != []:
                np_countdown_down[i] = max(
                    countdown_down_list
                )  # this is enough for our purposes

        # counter increase
        i += 1

    # Join np arrays with the dataframe
    setup_df = pd.DataFrame(data=np_setup)
    setup_df.index = bars.index.copy()
    bars["td_setup"] = setup_df

    setup_dir = pd.DataFrame(data=np_direction)
    setup_dir.index = bars.index.copy()
    bars["td_direction"] = setup_dir

    move_extreme_df = pd.DataFrame(data=np_move_extremes)
    move_extreme_df.index = bars.index.copy()
    bars["move_extreme"] = move_extreme_df

    np_countdown_up_df = pd.DataFrame(data=np_countdown_up)
    np_countdown_up_df.index = bars.index.copy()
    bars["countdown_up"] = np_countdown_up_df

    np_countdown_down_df = pd.DataFrame(data=np_countdown_down)
    np_countdown_down_df.index = bars.index.copy()
    bars["countdown_down"] = np_countdown_down_df

    # Adding tdst
    np_tdst_df = pd.DataFrame(data=np_tdst)
    np_tdst_df.index = bars.index.copy()
    bars["tdst"] = np_tdst_df

    # Change types to save memory
    bars["countdown_up"] = bars.countdown_up.astype("int8")
    bars["countdown_down"] = bars.countdown_down.astype("int8")
    bars["td_setup"] = bars.td_setup.astype("int8")
    bars["open"] = bars.open.astype("float32")
    bars["high"] = bars.high.astype("float32")
    bars["low"] = bars.low.astype("float32")
    bars["close"] = bars.close.astype("float32")
    # new
    bars["tdst"] = bars.tdst.astype("float32")

    bars["timestamp"] = df["timestamp"]

    # Returning only what we need
    return bars[
        [
            "timestamp",
            "td_setup",
            "td_direction",
            "move_extreme",
            "countdown_down",
            "countdown_up",
            "tdst",
        ]
    ]


__all__ = [
    "td_indicators",
    "ATR",
    "wwma",
    "MA",
    "RSI",
    "ROC",
    "streak",
    "CRSI",
    "StochRSI",
    "combined_indicators",
]
