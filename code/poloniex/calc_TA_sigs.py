import talib
import polo_eda as pe
import matplotlib.pyplot as plt

def create_tas(df):
    # create ohlc bars
    ticks = df.loc[:, ['rate', 'amount']].iloc[::-1]
    bars = ticks['rate'].resample('1min').ohlc()
    bars = bars.fillna(method='ffill')

    # bollinger bands
    # strange bug, if values are small, need to multiply to larger value for some reason
    mult = 1
    last_close = bars.iloc[0]['close']
    lc_m = last_close * mult
    while lc_m < 1:
        mult *= 10
        lc_m = last_close * mult

    print('using multiplier of', mult)

    upper, middle, lower = talib.BBANDS(bars['close'].values*mult,
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

    # note: too small of a timeperiod will result in junk data...I think.  or at least very discretized
    mom = talib.MOM(bars['close'].values*mult, timeperiod=14)

    bars['mom'] = mom/mult
    bars['mom'].fillna(method='bfill', inplace=True)

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
