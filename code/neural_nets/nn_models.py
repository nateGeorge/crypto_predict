import tensorflow as tf
import pandas as pd

def create_hist_feats_all(dfs, history=300, future=5):
    """
    Creates features from historical data, but creates features up to the most
    current date available.  Assumes dataframe goes from
    oldest date at .iloc[0] to newest date at .iloc[-1]
    :param dfs: list of dataframes with market/security as key
    :param history_days: number of points to use for prediction
    :param future_days: points in the future we want to predict for
    :returns:

    The default assumes the data has been resampled to OHLCV at minutely
    resolution.  So we are using 300 minutes (6 hours) to predict 3 hours in
    the future.
    """
    columns = ['open', 'high', 'low', 'close', 'volume']
    target_col = 'close'
    feats = {}
    targs = {}
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        # create time-lagged features
        features = []
        targets = []
        for i in range(history, data_points - future):
            features.append(dfs[s].iloc[i - history:i][columns].values)
            targets.append(dfs[s].iloc[i + future][target_col])

        feats[s] = np.array(features)
        # make the targets a column vector
        targs[s] = np.array(targets).reshape(-1, 1)

    return feats, targs


def transform_data(df, mva=30):
    """
    This is for scaling the original data for use in machine learning.

    Takes a numpy array as input, divides by a moving average with period mva (integer),
    and returns the scaled data as well as the scaler
    and moving average (needed for returning the data to its original form).

    :param df:
    :param mva: moving average period

    With minutely OHLCV bars, this takes a 30-min MVA by default.
    """
    scalers = {}
    new_df_dict = {}
    for c in df.columns:
        scalers[c] = SS()
        rolling_mean = df[c].rolling(window=mva).mean().bfill()
        mva_scaled = df[c] / rolling_mean
        # use sklearn scaler to fit and transform scaled values
        scaled = scalers[c].fit_transform(mva_scaled.values.reshape(-1, 1))
        # need to just grab values from the column for the dataframe to work
        new_df_dict[c] = scaled[:, 0]

    new_df = pd.DataFrame(new_df_dict)

    return new_df, scalers


def reform_data(df, scaled_df, scalers, mva=30):
    """
    Re-constructs original data from the transformed data.
    Be careful to make sure the mva is the same as used when scaling the data.
    """
    unsc_dict = {}
    for c in scaled_df.columns:
        unscaled = scalers[c].inverse_transform(scaled_df[c])
        rolling_mean = df[c].rolling(window=mva).mean().bfill()
        unsc = unscaled * rolling_mean
        unsc_dict[c] = unsc

    unsc_df = pd.DataFrame(unsc_dict)

    return unsc_df


def rescale_data(scaled_df, scalers):
    unsc_dict = {}
    for c in scaled_df.columns:
        unscaled = scalers[c].inverse_transform(scaled_df[c])
        unsc_dict[c] = unscaled

    unsc_df = pd.DataFrame(unsc_dict)

    return unsc_df


def rescale_prediction(prediction, scalers):
    """
    :param prediction: the closing price prediction, which has been scaled
    :param scalers: dict of standardscalers used to scale known data
    """
    return scalers['close'].inverse_transform(prediction.reshape(-1, 1))[0][0]


def unscale_prediction(df, prediction, mva=30):
    """
    :param df: a pandas dataframe of the known data.  This should be the natural data,
                not scaled or normalized
    :param prediction: rescaled
    """
    return df.iloc[-mva + 1:].close.sum() / (mva - prediction)
