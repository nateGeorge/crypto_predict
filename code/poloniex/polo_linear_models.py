# core
import os
import time
import sys

# custom
import polo_eda as pe
sys.path.append('../')
import data_processing as dp
import utils
import calc_TA_sigs as cts

# installed
import pandas as pd
import numpy as np
import pickle as pk
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as lr

MARKETS = pe.get_all_trade_pairs()
HOME_DIR = utils.get_home_dir()


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
        df = load_data(m)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        if rs_full.shape[0] < 2000:
            print('less than 2000 points for this market, skipping the model...')
            continue
        elif rs_full.shape[0] < 5000:
            cut = 2000

        rs_full = dp.make_mva_features(rs_full)
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


def load_data(m):
    notloaded = True
    sleep_time = 10 # start with 10s sleep time
    while notloaded:
        try:
            df = pe.read_trade_hist(m)
            notloaded = False
        except EOFError:  # this happens if the data is being written currently I think
            #...give it a bit to finish scraping
            print('EOF error, pausing for', str(sleep_time) + 's...')
            time.sleep(sleep_time)
            sleep_time += 10

    return df


def make_linear_models():
    """
    creates linear models for each currency pair
    uses statsmodels lib so we can have confidence intervals
    """
    hist = 12
    models = {}
    r2s = {}
    for m in sorted(MARKETS):
        cut = 5000
        print('loading', m + '...')
        df = load_data(m)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        if rs_full.shape[0] < 2000:
            print('less than 2000 points for this market, skipping the model...')
            continue
        elif rs_full.shape[0] < 5000:
            cut = 2000

        rs_full = dp.make_mva_features(rs_full)
        rs_full = cts.create_tas(bars=rs_full)
        # skip the first 'cut' points
        rs_full = rs_full.iloc[cut:]

        x_cols = get_xcols()
        X = sm.add_constant(rs_full[x_cols].iloc[24 - hist:-hist].values, has_constant='add')
        y = rs_full['24h_price_diff'].iloc[24:].values
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared_adj
        print('r^2 for', m, ':', '%.3f' % r2)
        models[m] = model
        r2s[m] = r2

    return models, r2s


def get_xcols():
    indicators = ['bband_u_cl', # bollinger bands
                 'bband_m_cl',
                 'bband_l_cl',
                 'bband_u_tp',
                 'bband_m_tp',
                 'bband_l_tp',
                 'dema_cl',
                 'dema_tp',
                 'ema_cl',
                 'ema_tp',
                 'ht_tl_cl',
                 'ht_tl_tp',
                 'kama_cl',
                 'kama_tp',
                #  'mama_cl',
                #  'mama_tp',
                #  'fama_cl',
                #  'fama_tp',
                #  'mama_cl_osc',
                #  'mama_tp_osc',
                 'midp_cl',
                 'midp_tp',
                 'midpr',
                 'sar',
                 'tema_cl',
                 'tema_tp',
                 'trima_cl',
                 'trima_tp',
                 'wma_cl',
                 'wma_tp',
                 'adx',
                 'adxr',
                 'apo_cl',
                 'apo_tp',
                 'arup', # aroon
                 'ardn',
                 'aroonosc',
                 'bop',
                 'cci',
                 'cmo_cl',
                 'cmo_tp',
                 'dx',
                 'macd_cl',
                 'macdsignal_cl',
                 'macdhist_cl',
                 'macd_tp',
                 'macdsignal_tp',
                 'macdhist_tp',
                 'mfi',
                 'mdi',
                 'mdm',
                 'mom_cl',
                 'mom_tp',
                 'pldi',
                 'pldm',
                 'ppo_tp',
                 'roc_cl',
                 'roc_tp',
                 'rocp_cl',
                 'rocp_tp',
                 'rocr_cl',
                 'rocr_tp',
                 'rocr_cl_100',
                 'rocr_tp_100',
                 'rsi_cl',
                 'rsi_tp',
                 'slowk', # stochastic oscillator
                 'slowd',
                 'fastk',
                 'fastd',
                 'strsi_cl_k',
                 'strsi_cl_d',
                 'strsi_tp_k',
                 'strsi_tp_d',
                 'trix_cl',
                 'trix_tp',
                 'ultosc',
                 'willr',
                 'ad',
                 'adosc',
                 'obv_cl',
                 'obv_tp',
                 'atr',
                 'natr',
                 'trange',
                 'ht_dcp_cl',
                 'ht_dcp_tp',
                 'ht_dcph_cl',
                 'ht_dcph_tp',
                 'ht_ph_cl',
                 'ht_ph_tp',
                 'ht_q_cl',
                 'ht_q_tp',
                 'ht_s_cl',
                 'ht_s_tp',
                 'ht_ls_cl',
                 'ht_ls_tp',
                 'ht_tr_cl',
                 'ht_tr_tp'
                 ]

    return indicators + ['mva_tp_24_diff', 'direction_volume']


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
            df = load_data(m)

            # resamples to the hour
            rs_full = dp.resample_ohlc(df, resamp='H')
            rs_full = dp.make_mva_features(rs_full)
            rs_full = cts.create_tas(bars=rs_full)
            last_price = rs_full.iloc[-1]['typical_price']
            cur_mva_diffs.append(rs_full.iloc[-1]['mva_tp_24_diff'])

            x_cols = get_xcols()
            X = sm.add_constant(rs_full[x_cols].iloc[-hist:].values, has_constant='add')
            preds = models[m].predict(X)
            scaled_preds.append(preds[-1] / last_price)
            print('scaled prediction:', '%.3f' % (scaled_preds[-1] * 100) + '%')
            pred_pairs.append(m)

    scaled_preds = np.array(scaled_preds)
    cur_mva_diffs = np.array(cur_mva_diffs)
    pred_pairs = np.array(pred_pairs)

    pred_idx = np.argsort(scaled_preds)[::-1]
    if scaled_preds[pred_idx][0] > 0:
        print('top 10 buys right now:')
        for i in pred_idx[:10]:
            print(pred_pairs[i] + ',', 'up', '%.1f' % (scaled_preds[i] * 100) + '%')
    else:
        print('nothing trending up right now!')

    pred_idx = np.argsort(scaled_preds)
    if scaled_preds[pred_idx][0] < 0:
        print('top 10 shorts right now:')
        for i in pred_idx[:10]:
            print(pred_pairs[i] + ',', 'down', '%.1f' % (scaled_preds[i] * 100) + '%')
    else:
        print('nothing trending down right now!')



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
    model_file = model_folder + 'models.pk'
    if os.path.exists(model_file):
        ms = pk.load(open(model_file, 'rb'))
        r2s = pk.load(open(model_folder + 'r2s.pk', 'rb'))
    else:
        make_and_save_models()
        ms = pk.load(open(model_file, 'rb'))
        r2s = pk.load(open(model_folder + 'r2s.pk', 'rb'))

    return ms, r2s


if __name__ == "__main__":
    ms, r2s = load_models_r2s()
    get_future_preds(ms, r2s)
