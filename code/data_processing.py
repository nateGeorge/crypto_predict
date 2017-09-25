import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SS


def resample_ohlc(df, resamp='T', vol_col='total'):
    """
    Currently it's resampling the btc amount (or first trading pair)
    as the volume.

    To use the second trading pair as the volume, change 'total' to
    'amount'.

    Other resampling times:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    :param df: pandas dataframe with 'amount', 'rate', and 'volume' in columns
    :param resamp: the time period to resample to; 'T' (default) is minutely
    :param vol_col: either 'total' or 'amount'.  'total' will use the
                    second currency (usually BTC) as the volume
                    calculator, 'amount' will use the first currency
                    in the pair

    :returns: pandas dataframe of resampled data with volume and
              directional volume
    """
    rs = df['rate'].resample(resamp).ohlc().interpolate()
    # calculate directional volume
    df['dir_mult'] = 1
    df.loc[df['type'] == 'sell', 'dir_mult'] = -1
    df['direction_volume'] = df['dir_mult'] * df[vol_col]
    rs_dir_amt = pd.DataFrame(df['direction_volume'].resample(resamp).sum().interpolate())
    rs_vol = pd.DataFrame(df[vol_col].resample(resamp).sum().interpolate())
    rs_vol.rename(columns={vol_col: 'volume'}, inplace=True)
    rs_full = rs.merge(rs_vol, left_index=True, right_index=True)
    rs_full = rs_full.merge(rs_dir_amt, left_index=True, right_index=True)
    return rs_full


def transform_data(df, mva=30):
    """
    WARNING: have not tried with directional volume yet, guessing
    that there will be problems.

    This is for scaling the original data for use in machine learning.

    Takes a numpy array as input, divides by a moving average with period mva (integer),
    and returns the scaled data as well as the scaler
    and moving average (needed for returning the data to its original form).

    :param df:
    :param mva: moving average period, pass 'None' to not use mva
                scaling

    With minutely OHLCV bars, this takes a 30-min MVA by default.
    """
    scalers = {}
    new_df_dict = {}
    for c in df.columns:
        scalers[c] = SS()
        if mva is not None:
            rolling_mean = df[c].rolling(window=mva).mean().bfill()
            mva_scaled = df[c] / rolling_mean
            # use sklearn scaler to fit and transform scaled values
            scaled = scalers[c].fit_transform(mva_scaled.values.reshape(-1, 1))
            # need to just grab values from the column for the dataframe to work
            new_df_dict[c] = scaled[:, 0]
        else:
            new_df_dict[c] = scalers[c].fit_transform(df[c].values.reshape(-1, 1))[:, 0]

    new_df = pd.DataFrame(new_df_dict)
    new_df.set_index(df.index, inplace=True)

    return new_df, scalers


def reform_data(df, scaled_df, scalers, mva=30):
    """
    Re-constructs original data from the transformed data.
    Be careful to make sure the mva is the same as used when scaling the data.
    """
    unsc_dict = {}
    for c in scaled_df.columns:
        unscaled = scalers[c].inverse_transform(scaled_df[c])
        if mva is not None:
            if scaled_df.shape[0] <= mva:
                rolling_mean = df.iloc[-(scaled_df.shape[0] + mva):][c].rolling(window=mva).mean().bfill()
                rolling_mean = rolling_mean[-scaled_df.shape[0]:]
            else:
                rolling_mean = df[c].rolling(window=mva).mean().bfill()
                if rolling_mean.shape[0] > unscaled.shape[0]:
                    rolling_mean = rolling_mean[-unscaled.shape[0]:]

            unsc = unscaled * rolling_mean
            unsc_dict[c] = unsc
        else:
            unsc_dict[c] = unscaled

    unsc_df = pd.DataFrame(unsc_dict)

    return unsc_df


def rescale_data(scaled_df, scalers):
    unsc_dict = {}
    for c in scaled_df.columns:
        unscaled = scalers[c].inverse_transform(scaled_df[c])
        unsc_dict[c] = unscaled

    unsc_df = pd.DataFrame(unsc_dict)

    return unsc_df


def inv_xform_prediction(prediction, scalers):
    """
    :param prediction: the closing price prediction, which has been scaled
    :param scalers: dict of standardscalers used to scale known data
    """
    return scalers['close'].inverse_transform(prediction.reshape(-1, 1))[0][0]


def reform_prediction(df, prediction, mva=30):
    """
    This was really intended only to work as the very last point in the
    known data.  So if using intermediate points, you have to cut off the df
    at the last point before it's being predicted.
    :param df: a pandas dataframe of the known data.  This should be the natural data,
                not scaled or normalized
    :param prediction: rescaled
    """
    return df.iloc[-mva + 1:].close.sum() / (mva / prediction - 1)


def reform_future_predictions(df, preds, mva=30):
    """
    rescales future predictions
    :param df: pandas dataframe of current data, up to most recent timestep before future predictions
    :param preds: numpy array of predictions, with shape (x,), should be rescaled (i.e. all positive)
    :param mva: integeter specifying moving average period (same as used for scaling the data)
    """
    resc_preds = [reform_prediction(df, preds[0], mva=mva)]
    # optional: set the first point to the last known point
    # resc_preds = [df.close[-mva]]
    for i in range(1, mva - 1):
        cur = (df.iloc[-mva + i + 1:].close.sum() + sum(resc_preds)) / (mva / preds[i] - 1)
        resc_preds.append(cur)

    for i in range(mva - 1, preds.shape[0]):
        cur = sum(resc_preds[-mva + i + 1:i]) / (mva / preds[i] - 1)
        resc_preds.append(cur)

    return pd.DataFrame({'close': resc_preds})


def reform_future_predictions_mild(df, preds, mva=30):
    """
    rescales future predictions, but uses the last mva of the actual data
    instead of the mva of the predictions

    unfortunately, this seems to make the predictions just flatline...
    :param df: pandas dataframe of current data, up to most recent timestep before future predictions
    :param preds: numpy array of predictions, with shape (x,), should be rescaled (i.e. all positive)
    :param mva: integeter specifying moving average period (same as used for scaling the data)
    """
    resc_preds = [reform_prediction(df, preds[0], mva=mva)]
    # optional: set the first point to the last known point
    # resc_preds = [df.close[-mva]]
    for i in range(1, preds.shape[0]):
        cur = (df.iloc[-mva + 2:].close.sum() + resc_preds[0]) / (mva / preds[i] - 1)
        resc_preds.append(cur)

    return pd.DataFrame({'close': resc_preds})


def create_hist_feats(df, history=300, future=5):
    """
    Creates features from historical data, but creates features up to the most
    current date available.  Assumes dataframe goes from
    oldest date at .iloc[0] to newest date at .iloc[-1]
    :param df: pandas dataframe with trade data
    :param history_days: number of points to use for prediction
    :param future_days: points in the future we want to predict for
    :returns:

    The default assumes the data has been resampled to OHLCV at minutely
    resolution.  So we are using 300 minutes (6 hours) to predict 3 hours in
    the future.
    """
    columns = ['open', 'high', 'low', 'close', 'volume', 'direction_volume']
    target_col = 'close'
    data_points = df.shape[0]
    # create time-lagged features
    features = []
    targets = []
    for i in range(history, data_points - future):
        features.append(df.iloc[i - history:i][columns].values)
        targets.append(df.iloc[i + future][target_col])

    feats = np.array(features)
    # make the targets a column vector
    targs = np.array(targets).reshape(-1, 1)

    return feats, targs


def create_feats_to_current(df, history=300, future=5):
    """
    Same as create_hist_feats, but creates features all the way to the most
    current timestep.
    """
    columns = ['open', 'high', 'low', 'close', 'volume', 'direction_volume']
    target_col = 'close'
    data_points = df.shape[0]
    # create time-lagged features
    features = []
    targets = []
    for i in range(history, data_points - future):
        features.append(df.iloc[i - history:i][columns].values)
        targets.append(df.iloc[i + future][target_col])

    for i in range(data_points - future, data_points):
        features.append(df.iloc[i - history:i][columns].values)

    feats = np.array(features)
    # make the targets a column vector
    targs = np.array(targets).reshape(-1, 1)

    return feats, targs
