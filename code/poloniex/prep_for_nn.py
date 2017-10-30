"""
prepares data for neural network models (10/2017)

meant to be run from the home code dir
"""
# core
import os
import sys

# custom
sys.path.append('code')
sys.path.append('code/poloniex')
import polo_eda as pe
import calc_TA_sigs as cts
import data_processing as dp
from utils import get_home_dir

# installed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler as SS
import h5py
import numpy as np
from poloniex import Poloniex
import pickle as pk

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)

# yeah yeah, should be caps
indicators = ['bband_u_cl', # bollinger bands
             'bband_m_cl',
             'bband_l_cl',
             'bband_u_tp',
             'bband_m_tp',
             'bband_l_tp',
             'bband_u_cl_diff',
             'bband_m_cl_diff',
             'bband_l_cl_diff',
             'bband_u_cl_diff_hi',
             'bband_l_cl_diff_lo',
             'bband_u_tp_diff',
             'bband_m_tp_diff',
             'bband_l_tp_diff',
             'bband_u_tp_diff_hi',
             'bband_l_tp_diff_lo',
             'dema_cl',
             'dema_tp',
             'dema_cl_diff',
             'dema_tp_diff',
             'ema_cl',
             'ema_tp',
             'ema_cl_diff',
             'ema_tp_diff',
             'ht_tl_cl',
             'ht_tl_tp',
             'ht_tl_cl_diff',
             'ht_tl_tp_diff',
             'kama_cl',
             'kama_tp',
             'kama_cl_diff',
             'kama_tp_diff',
            #  'mama_cl',  # having problems with these
            #  'mama_tp',
            #  'fama_cl',
            #  'fama_tp',
            #  'mama_cl_osc',
            #  'mama_tp_osc',
             'midp_cl',
             'midp_tp',
             'midp_cl_diff',
             'midp_tp_diff',
             'midpr',
             'midpr_diff',
             'sar',
             'sar_diff',
             'tema_cl',
             'tema_tp',
             'tema_cl_diff',
             'tema_tp_diff',
             'trima_cl',
             'trima_tp',
             'trima_cl_diff',
             'trima_tp_diff',
             'wma_cl',
             'wma_tp',
             'wma_cl_diff',
             'wma_tp_diff',
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
             'ppo_cl',
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
# home_dir = get_home_dir()  # data too big for this right now
home_dir = '/media/nate/data_lake/crytpo_predict/'


def make_data_dirs():
    for d in [home_dir + 'data', home_dir + 'data/nn_feats_targs', home_dir + 'data/nn_feats_targs/poloniex']:
        if not os.path.exists(d):
            os.mkdir(d)


def get_btc_usdt_pairs():
    """
    filters out eth/xmr primary trading pairs
    """
    # gets all markets
    ticks = polo.returnTicker()
    pairs = sorted(ticks.keys())
    # for now, just BTC and USDT are maximized
    btc_pairs = [p for p in pairs if 'BTC' == p[:3]]
    usdt_pairs = [p for p in pairs if 'USDT' == p[:4]]
    pairs = btc_pairs + usdt_pairs
    return pairs


def make_all_nn_data(make_fresh=False, skip_load=True, save_scalers=False):
    ticks = polo.returnTicker()
    pairs = sorted(ticks.keys())
    # for now, just care about BTC and USDT
    btc_pairs = [p for p in pairs if 'BTC' == p[:3]]
    usdt_pairs = [p for p in pairs if 'USDT' == p[:4]]
    pairs = btc_pairs + usdt_pairs
    for c in pairs:
        print('making data for', c)
        _, _, _, _ = prep_polo_nn(mkt=c, make_fresh=make_fresh, skip_load=skip_load, save_scalers=save_scalers)


def prep_polo_nn(mkt='BTC_AMP', make_fresh=False, skip_load=False, save_scalers=False):
    """
    right now this is predicting 24h into the future.  need to make future time arbitrary
    :param mkt: string, market pair to use
    :param make_fresh: creates new files even if they exist
    :param skip_load: if files already exist, just returns all Nones
    :param save_scalers: will save the StandardScalers used.  necessary to do
                        live predictions
    """
    datafile = home_dir + 'data/nn_feats_targs/poloniex/' + mkt
    if os.path.exists(datafile) and not make_fresh:
        if skip_load:
            print('files exist, skipping')
            return None, None, None, None

        print('loading...')
        f = h5py.File(datafile, 'r')
        xform_train = f['xform_train'][:]
        xform_test = f['xform_test'][:]
        train_targs = f['train_targs'][:]
        test_targs = f['test_targs'][:]
        f.close()
    else:
        print('creating new...')
        df = pe.read_trade_hist(mkt)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        rs_full = dp.make_mva_features(rs_full)
        bars = cts.create_tas(bars=rs_full, verbose=True)
        # make target columns
        col = '24h_price_diff'
        bars[col] = bars['typical_price'].copy()
        bars[col] = np.hstack((np.repeat(bars[col].iloc[24], 24), bars['typical_price'].iloc[24:].values - bars['typical_price'].iloc[:-24].values))
        bars['24h_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[24], 24), bars['typical_price'].iloc[24:].values))
        # drop first 24 points because they are repeated
        bars = bars.iloc[24:]
        # also drop first 1000 points because usually they are bogus
        bars = bars.iloc[1000:]
        if bars.shape[0] < 1000:
            print('less than 1000 points, skipping...')
            return None, None, None, None

        feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume']
        features = bars[feat_cols].values

        new_feats, targets = create_hist_feats(features, bars, hist_points=480)
        xform_train, xform_test, train_targs, test_targs = scale_historical_feats(new_feats, targets, test_size=5000)

        f = h5py.File(datafile, 'w')
        f.create_dataset('xform_train', data=xform_train, compression='lzf')
        f.create_dataset('xform_test', data=xform_test, compression='lzf')
        f.create_dataset('train_targs', data=train_targs, compression='lzf')
        f.create_dataset('test_targs', data=test_targs, compression='lzf')
        f.close()

    return xform_train, xform_test, train_targs, test_targs


def make_all_polo_nn_fulltrain(skip_load=True):
    pairs = get_btc_usdt_pairs()
    start = pairs.index('BTC_ETH')
    for p in pairs[start:]:
        print('making full train for', p)
        _, _ = make_polo_nn_fulltrain(p, skip_load=skip_load)


def make_polo_nn_fulltrain(mkt='BTC_AMP', make_fresh=False, skip_load=False):
    datafile = home_dir + 'data/nn_feats_targs/poloniex/full_train_' + mkt
    if os.path.exists(datafile) and not make_fresh:
        if skip_load:
            print('skipping...')
            return None, None

        print('loading...')
        f = h5py.File(datafile, 'r')
        xform_train = f['xform_train'][:]
        train_targs = f['train_targs'][:]
        f.close()
    else:
        print('creating new...')
        df = pe.read_trade_hist(mkt)
        # resamples to the hour
        rs_full = dp.resample_ohlc(df, resamp='H')
        rs_full = dp.make_mva_features(rs_full)
        bars = cts.create_tas(bars=rs_full, verbose=True)
        # make target columns
        col = '24h_price_diff'
        bars[col] = bars['typical_price'].copy()
        bars[col] = np.hstack((np.repeat(bars[col].iloc[24], 24), bars['typical_price'].iloc[24:].values - bars['typical_price'].iloc[:-24].values))
        bars['24h_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[24], 24), bars['typical_price'].iloc[24:].values))
        # drop first 24 points because they are repeated
        bars = bars.iloc[24:]
        # also drop first 1000 points because usually they are bogus
        bars = bars.iloc[1000:]
        if bars.shape[0] < 1000:
            print('less than 1000 points, skipping...')
            return None, None

        feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume']
        features = bars[feat_cols].values

        new_feats, train_targs = create_hist_feats(features, bars, hist_points=480)
        xform_train = scale_historical_feats_full(mkt, new_feats)

        f = h5py.File(datafile, 'w')
        f.create_dataset('xform_train', data=xform_train, compression='lzf')
        f.create_dataset('train_targs', data=train_targs, compression='lzf')
        f.close()

    return xform_train, train_targs


def create_hist_feats(features, bars, hist_points=480):
    # make historical features
    new_feats = []
    for i in range(hist_points, features.shape[0]):
        new_feats.append(features[i - hist_points:i, :])

    new_feats = np.array(new_feats)
    targets = bars['24h_price_diff_pct'].iloc[hist_points:].values
    return new_feats, targets


def scale_historical_feats(new_feats, targets, test_size=5000, test_frac=0.2):
    # also makes train/test
    # in case dataset is too small for 5k test points, adjust according to test_frac
    if new_feats.shape[0] * test_frac < test_size:
        test_size = int(new_feats.shape[0] * test_frac)

    train_feats = new_feats[:-test_size]
    train_targs = targets[:-test_size]
    test_feats = new_feats[-test_size:]
    test_targs = targets[-test_size:]

    xform_train = []
    xform_test = []
    s = SS()
    for j in tqdm(range(train_feats.shape[0])):  # timesteps
        xform_train_ts = []
        for i in range(train_feats.shape[2]):  # number of indicators/etc
            xform_train_ts.append(s.fit_transform(train_feats[j, :, i].reshape(-1, 1))[:, 0])

        xform_train_ts = np.array(xform_train_ts).reshape(train_feats.shape[1], train_feats.shape[2])
        xform_train.append(xform_train_ts)

    for j in tqdm(range(test_feats.shape[0])):
        xform_test_ts = []
        for i in range(test_feats.shape[2]):  # number of indicators/etc
            xform_test_ts.append(s.fit_transform(test_feats[j, :, i].reshape(-1, 1))[:, 0])

        xform_test_ts = np.array(xform_test_ts).reshape(test_feats.shape[1], test_feats.shape[2])
        xform_test.append(xform_test_ts)

    print(train_feats.shape[0])
    xform_train = np.array(xform_train).reshape(train_feats.shape[0], train_feats.shape[1], train_feats.shape[2])
    xform_test = np.array(xform_test).reshape(test_feats.shape[0], test_feats.shape[1], test_feats.shape[2])

    return xform_train, xform_test, train_targs, test_targs


def scale_historical_feats_full(mkt, train_feats):
    xform_train = []
    scalers = [SS() for i in range(train_feats.shape[2])]
    for j in tqdm(range(train_feats.shape[0])):  # timesteps
        xform_train_ts = []
        for i in range(train_feats.shape[2]):  # number of indicators/etc
            xform_train_ts.append(scalers[i].fit_transform(train_feats[j, :, i].reshape(-1, 1))[:, 0])

        xform_train_ts = np.array(xform_train_ts).reshape(train_feats.shape[1], train_feats.shape[2])
        xform_train.append(xform_train_ts)

    print(train_feats.shape[0])
    xform_train = np.array(xform_train).reshape(train_feats.shape[0], train_feats.shape[1], train_feats.shape[2])

    sc_file = home_dir + 'data/nn_feats_targs/poloniex/full_train_' + mkt + '_scalers.pk'
    pk.dump(scalers, open(sc_file, 'wb'))  # make sure to change this to read after copy-pasting!

    return xform_train
