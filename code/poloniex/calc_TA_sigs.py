import numpy as np
import talib
import polo_eda as pe
import matplotlib.pyplot as plt

def create_tas(df=None, bars=None, col='close'):
    """
    :param col: the column to use for creating TAs

    idea for future: instead of just one col for the calcs, use
    close and typical price ([close + high + low] / 3)

    list of currently applied indicators:
    ['bband_u', # bollinger bands
     'bband_m',
     'bband_l',
     'dema',
     'ema',
     'midp',
     'sar',
     'tema',
     'trima',
     'wma',
     'ad',
     'adosc',
     'obv',
     'trange',
     'mom',
     'apo',
     'arup', # aroon
     'ardn'
     'aroonosc',
     'bop',
     'cci',
     'macd',
     'macdsignal',
     'macdhist',
     'ppo',
     'slowk', # stochastic oscillator
     'slowd',
     'fastk',
     'fastd',
     'ultosc',
     'willr'
     ]
    """
    if bars is None:
        # create ohlc bars -- old way, there is another way in
        ticks = df.loc[:, ['rate', 'amount']].iloc[::-1]
        bars = ticks['rate'].resample('1min').ohlc()
        bars = bars.fillna(method='ffill')

    # bollinger bands
    # strange bug, if values are small, need to multiply to larger value for some reason
    mult = 1
    last_close = bars.iloc[0][col]
    lc_m = last_close * mult
    while lc_m < 1:
        mult *= 10
        lc_m = last_close * mult

    print('using multiplier of', mult)
    mult_col = bars[col].values * mult
    mult_close = bars['close'].values * mult
    mult_open = bars['open'].values * mult
    mult_high = bars['high'].values * mult
    mult_low = bars['low'].values * mult


    ### overlap studies
    upper, middle, lower = talib.BBANDS(mult_col,
                                    timeperiod=10,
                                    nbdevup=2,
                                    nbdevdn=2)
    upper /= mult
    middle /= mult
    lower /= mult
    bars['bband_u'] = upper
    bars['bband_m'] = middle
    bars['bband_l'] = lower
    bars['bband_u'].fillna(method='bfill', inplace=True)
    bars['bband_m'].fillna(method='bfill', inplace=True)
    bars['bband_l'].fillna(method='bfill', inplace=True)

    # Double Exponential Moving Average
    bars['dema'] = talib.DEMA(mult_col, timeperiod=30) / mult

    # exponential moving Average
    bars['ema'] = talib.EMA(mult_col, timeperiod=30) / mult

    # Moving average with variable period
    bars['mavp'] = talib.MAVP(mult_close, np.arange(mult_close.shape[0]).astype(np.float64), minperiod=2, maxperiod=30, matype=0) / mult

    # midpoint price over period
    bars['midp'] = talib.MIDPRICE(mult_high, mult_low, timeperiod=14)

    # parabolic sar
    bars['sar'] = talib.SAR(mult_high, mult_low, acceleration=0, maximum=0)
    # need to make an oscillator for this

    # triple exponential moving average
    bars['tema'] = talib.TEMA(mult_col, timeperiod=30)

    # triangular ma
    bars['trima'] = talib.TRIMA(mult_col, timeperiod=30)

    # weighted moving average
    bars['wma'] = talib.WMA(mult_col, timeperiod=30)


    ### volume indicators
    # Chaikin A/D Line
    bars['ad'] = talib.AD(mult_high, mult_low, mult_close, bars['volume'].values)

    # Chaikin A/D Oscillator
    bars['adosc'] = talib.ADOSC(mult_high, mult_low, mult_close, bars['volume'].values, fastperiod=3, slowperiod=10)

    # on balance volume
    bars['obv'] = talib.OBV(mult_col, bars['volume'].values)


    ### volatility indicators
    # true range
    bars['trange'] = talib.TRANGE(mult_high, mult_low, mult_close) / mult


    #### momentum indicators  -- for now left out those with unstable periods
    # note: too small of a timeperiod will result in junk data...I think.  or at least very discretized
    mom = talib.MOM(mult_col, timeperiod=14)

    bars['mom'] = mom / mult
    bars['mom'].fillna(method='bfill', inplace=True)

    # Absolute Price Oscillator
    # values around -100 to +100
    bars['apo'] = talib.APO(mult_col, fastperiod=12, slowperiod=26, matype=0)

    # Aroon and Aroon Oscillator 0-100, so don't need to renormalize
    arup, ardn = talib.AROON(mult_high, mult_low, timeperiod=14)
    bars['arup'] = arup
    bars['ardn'] = ardn

    # linearly related to aroon, just aroon up - aroon down
    bars['aroonosc'] = talib.AROONOSC(mult_high, mult_low, timeperiod=14)

    # balance of power - ratio of values so don't need to re-normalize
    bars['bop'] = talib.BOP(mult_open, mult_high, mult_low, mult_close)

    # Commodity Channel Index
    # around -100 to + 100
    bars['cci'] = talib.CCI(mult_high, mult_low, mult_close, timeperiod=14)

    # Moving Average Convergence/Divergence
    # https://www.quantopian.com/posts/how-does-the-talib-compute-macd-why-the-value-is-different
    # macd diff btw fast and slow EMA
    macd, macdsignal, macdhist = talib.MACD(mult_col, fastperiod=12, slowperiod=26, signalperiod=9)
    bars['macd'] = macd / mult
    bars['macdsignal'] = macdsignal / mult
    bars['macdhist'] = macdhist / mult

    # percentage price Oscillator
    bars['ppo'] = talib.PPO(mult_col, fastperiod=12, slowperiod=26, matype=0)

    # stochastic oscillator - % of price diffs, so no need to rescale
    slowk, slowd = talib.STOCH(mult_high, mult_low, mult_close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    fastk, fastd = talib.STOCHF(mult_high, mult_low, mult_close, fastk_period=5, fastd_period=3, fastd_matype=0)
    bars['slowk'] = slowk
    bars['slowd'] = slowd
    bars['fastk'] = fastk
    bars['fastd'] = fastd

    # ultimate Oscillator - between 0 and 100
    bars['ultosc'] = talib.ULTOSC(mult_high, mult_low, mult_close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # williams % r  -- 0 to 100
    bars['willr'] = talib.WILLR(mult_high, mult_low, mult_close, timeperiod=14)

    return bars


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return sr[ (sr - median).abs() <= iqr]

def remove_outliers(df):
    """
    removes outliers for EDA
    """
    data = {}
    for c in df.columns:
        print(c)
        data[c] = reject_outliers(df[c])

    return data


if __name__ == "__main__":
    df = pe.read_trade_hist()
