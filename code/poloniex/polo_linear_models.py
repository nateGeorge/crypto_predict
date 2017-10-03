# core
import os
import time
import sys

# custom
import polo_eda as pe
sys.path.append('../')
import data_processing as dp
import utils

# installed
import pandas as pd
import numpy as np
import pickle as pk
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as lr

MARKETS = pe.get_all_trade_pairs()
HOME_DIR = utils.get_home_dir()


def make_features(rs):
    """
    makes moving average features and price diff target for linear regression

    :param rs: resampled dataframe
    """
    # 24-hour moving average of typical price
    rs['mva_tp_24'] = rs['typical_price'].rolling(24).mean().bfill()
    # calculate slope/derivative of 24h mva
    rs['mva_tp_24_diff'] = rs['mva_tp_24'].diff().bfill()
    # make target, the 24 hour price difference
    rs['24h_price_diff'] = rs['typical_price'].copy()
    rs['24h_price_diff'].iloc[24:] = rs['typical_price'].iloc[24:].copy().values - rs['typical_price'].iloc[:-24].copy().values
    rs['24h_price_diff'].iloc[:24] = rs['24h_price_diff'].iloc[24].copy()

    return rs


def make_linear_models_sklearn():
    """
    creates linear models for each currency pair
    uses sklean lib, which makes it hard to get confidence intervals, etc
    """
    hist = 12
    models = {}
    r2s = {}
    for m in MARKETS:
        cut = 5000
        print('loading', m + '...')
        df = pe.read_trade_hist(m)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        if rs_full.shape[0] < 2000:
            print('less than 2000 points for this market, skipping the model...')
            continue
        elif rs_full.shape[0] < 5000:
            cut = 2000

        rs_full = make_features(rs_full)
        # skip the first 'cut' points
        rs_full = rs_full.iloc[cut:]

        X = rs_full[['mva_tp_24_diff',
                    'direction_volume']].iloc[24 - hist:-hist].values
        y = rs_full['24h_price_diff'].iloc[24:].values
        model = lr()
        model.fit(X, y)
        r2 = model.score(X, y)
        print('r^2 for', m, ':', '%.3f' % r2)
        models[m] = model
        r2s[m] = r2

    return models, r2s


def make_linear_models():
    """
    creates linear models for each currency pair
    uses statsmodels lib so we can have confidence intervals
    """
    hist = 12
    models = {}
    r2s = {}
    for m in MARKETS:
        cut = 5000
        print('loading', m + '...')
        df = pe.read_trade_hist(m)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        if rs_full.shape[0] < 2000:
            print('less than 2000 points for this market, skipping the model...')
            continue
        elif rs_full.shape[0] < 5000:
            cut = 2000

        rs_full = make_features(rs_full)
        # skip the first 'cut' points
        rs_full = rs_full.iloc[cut:]

        X = X = sm.add_constant(rs_full[['mva_tp_24_diff',
                                'direction_volume']].iloc[24 - hist:-hist].values)
        y = rs_full['24h_price_diff'].iloc[24:].values
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared
        print('r^2 for', m, ':', '%.3f' % r2)
        models[m] = model
        r2s[m] = r2

    return models, r2s


def get_future_preds(models, r2s):
    """
    gets prediction for the next 12 hours for all markets, and if any have a
    big enough move up, print an alert.
    """
    hist = 12
    scaled_preds = []
    pred_pairs = []
    cur_mva_diffs = []
    trading_pairs = sorted(models.keys())
    for m in trading_pairs:
        if r2s[m] >= 0.1:  # ignore the really poor models
            print('loading', m + '...')
            try:
                df = pe.read_trade_hist(m)
            except EOFError:  # this happens if the data is being written currently
                #...give it a bit to finish scraping
                time.sleep(60)
                df = pe.read_trade_hist(m)

            # resamples to the hour
            rs_full = dp.resample_ohlc(df, resamp='H')
            rs_full = make_features(rs_full)
            last_price = rs_full.iloc[-1]['typical_price']
            cur_mva_diffs.append(rs_full.iloc[-1]['mva_tp_24_diff'])
            X = rs_full[['mva_tp_24_diff',
                        'direction_volume']].iloc[-hist:].values
            preds = models[m].predict(X)
            scaled_preds.append(preds[-1] / last_price)
            print('scaled prediction:', str(preds[-1] / last_price))
            pred_pairs.append(m)

    scaled_preds = np.array(scaled_preds)
    cur_mva_diffs = np.array(cur_mva_diffs)
    pred_pairs = np.array(pred_pairs)

    pred_idx = np.argsort(scaled_preds)[::-1]
    if scaled_preds[pred_idx][0] > 0:
        print('top 3 buys right now:')
        for i in pred_idx[:3]:
            print(pred_pairs[i] + ',', 'up', '%.1f' % (scaled_preds[i] * 100) + '%')
    else:
        print('nothing trending up right now!')

    avg_trend = scaled_preds.mean()
    if avg_trend < 0:
        print('average market trend is DOWN:', '%.1f' % (avg_trend * 100) + '%')
    else:
        print('average market trend is UP:', '%.1f' % (avg_trend * 100) + '%')


def make_and_save_models():
    """
    makes models for all markets and saves them, along with r2 score
    """
    model_folder = HOME_DIR + 'models/'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    model_folder = model_folder + 'poloniex/'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    ms, r2s = make_linear_models()
    pk.dump(ms, open(model_folder + 'models.pk', 'wb'), pk.HIGHEST_PROTOCOL)
    pk.dump(r2s, open(model_folder + 'r2s.pk', 'wb'), pk.HIGHEST_PROTOCOL)


def load_models_r2s():
    """
    loads models and r-squared for all available markets
    """
    model_folder = HOME_DIR + 'models/poloniex/'
    ms = pk.load(open(model_folder + 'models.pk', 'rb'))
    r2s = pk.load(open(model_folder + 'r2s.pk', 'rb'))
    return ms, r2s


if __name__ == "__main__":
    ms, r2s = load_models_r2s()
    get_future_preds(ms, r2s)
