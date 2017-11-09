"""
prepares data for neural network models (10/2017)

meant to be run from the home code dir
"""
# core
import os
import sys
import gc

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
from multiprocessing import Pool
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot

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


def make_all_nn_data(future, hist_points, resamp, make_fresh=False, skip_load=True, save_scalers=False):
    ticks = polo.returnTicker()
    pairs = sorted(ticks.keys())
    # for now, just care about BTC and USDT
    btc_pairs = [p for p in pairs if 'BTC' == p[:3]]
    usdt_pairs = [p for p in pairs if 'USDT' == p[:4]]
    pairs = btc_pairs + usdt_pairs
    for c in pairs:
        print('making data for', c)
        _, _, _, _ = prep_polo_nn(mkt=c,
                                    make_fresh=make_fresh,
                                    skip_load=skip_load,
                                    save_scalers=save_scalers,
                                    future=future,
                                    hist_points=hist_points,
                                    resamp=resamp)


def prep_polo_nn(mkt='BTC_AMP', make_fresh=False, skip_load=False, save_scalers=False, future=24, hist_points=480, resamp='H'):
    """
    :param mkt: string, market pair to use
    :param make_fresh: creates new files even if they exist
    :param skip_load: if files already exist, just returns all Nones
    :param save_scalers: will save the StandardScalers used.  necessary to do
                        live predictions
    """
    datafile = make_filename(mkt, future, resamp, hist_points)
    print('filename:', datafile)
    if os.path.exists(datafile) and not make_fresh:
        if skip_load:
            print('files exist, skipping')
            return None, None, None, None

        print('loading...')
        f = h5py.File(datafile, 'r')
        train_feats = f['xform_train'][:]
        test_feats = f['xform_test'][:]
        train_targs = f['train_targs'][:]
        test_targs = f['test_targs'][:]
        dates = f['dates'][:]
        f.close()
    else:
        print('creating new...')
        df = pe.read_trade_hist(mkt)
        # resamples to the hour if H, T is for minutes, S is seconds
        rs_full = dp.resample_ohlc(df, resamp=resamp)
        del df
        gc.collect()
        rs_full = dp.make_mva_features(rs_full)
        bars = cts.create_tas(bars=rs_full, verbose=True)
        del rs_full
        gc.collect()
        # make target columns
        col = str(future) + '_' + resamp + '_price_diff'
        bars[col] = bars['typical_price'].copy()
        bars[col] = np.hstack((np.repeat(bars[col].iloc[future], future), bars['typical_price'].iloc[future:].values - bars['typical_price'].iloc[:-future].values))
        bars[str(future) + '_' + resamp + '_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[future], future), bars['typical_price'].iloc[:-future].values))
        # drop first 'future' points because they are repeated
        # also drop first 1000 points because usually they are bogus
        bars = bars.iloc[future + 1000:]
        dates = list(map(lambda x: x.value, bars.index))  # in microseconds since epoch
        if bars.shape[0] < 1000:
            print('less than 1000 points, skipping...')
            return None, None, None, None

        feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume', 'high', 'low', 'close', 'open']
        features = bars[feat_cols].values
        targets = bars[str(future) + '_' + resamp + '_price_diff_pct'].values
        del bars
        gc.collect()

        new_feats, targets, dates = create_hist_feats(features, targets, dates, hist_points=hist_points, future=future)
        test_size=5000
        test_frac=0.2
        scale_historical_feats(new_feats)  # does scaling in-place, no returned values
        # in case dataset is too small for 5k test points, adjust according to test_frac
        if new_feats.shape[0] * test_frac < test_size:
            test_size = int(new_feats.shape[0] * test_frac)

        train_feats = new_feats[:-test_size]
        train_targs = targets[:-test_size]
        test_feats = new_feats[-test_size:]
        test_targs = targets[-test_size:]

        # del targets
        # del new_feats
        # gc.collect()

        f = h5py.File(datafile, 'w')
        f.create_dataset('xform_train', data=train_feats, compression='lzf')
        f.create_dataset('xform_test', data=test_feats, compression='lzf')
        f.create_dataset('train_targs', data=train_targs, compression='lzf')
        f.create_dataset('test_targs', data=test_targs, compression='lzf')
        f.create_dataset('dates', data=dates, compression='lzf')
        f.close()

    return train_feats, test_feats, train_targs, test_targs, dates


def make_all_polo_nn_fulltrain(make_fresh=False, skip_load=True):
    pairs = get_btc_usdt_pairs()
    for p in pairs:
        print('making full train for', p)
        _, _ = make_polo_nn_fulltrain(p, make_fresh=make_fresh, skip_load=skip_load)


def make_filename(mkt, future, resamp, hist_points, full_train=False):
    if full_train:
        return home_dir + 'data/nn_feats_targs/poloniex/full_train_' + '_'.join([mkt, resamp, 'h=' + str(hist_points), 'f=' + str(future)])

    return home_dir + 'data/nn_feats_targs/poloniex/' + '_'.join([mkt, resamp, 'h=' + str(hist_points), 'f=' + str(future)])


def make_polo_nn_fulltrain(mkt='BTC_AMP', make_fresh=False, skip_load=False, hist_points=480, future=24, resamp='5T'):
    datafile = make_filename(mkt, future, resamp, hist_points, full_train=True)
    print('filename:', datafile)
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
        # make target columns -- price change from 24 hours ago
        col = str(future) + '_' + resamp + '_price_diff'
        bars[col] = bars['typical_price'].copy()
        bars[col] = np.hstack((np.repeat(bars[col].iloc[future], future), bars['typical_price'].iloc[future:].values - bars['typical_price'].iloc[:-future].values))
        bars[str(future) + '_' + resamp + '_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[future], future), bars['typical_price'].iloc[future:].values))
        # drop first 24 points because they are repeated
        bars = bars.iloc[future:]
        # also drop first 1000 points because usually they are bogus
        bars = bars.iloc[1000:]
        if bars.shape[0] < 1000:
            print('less than 1000 points, skipping...')
            return None, None

        feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume', 'high', 'low', 'close', 'open']
        features = bars[feat_cols].values
        targets = bars[str(future) + 'h_price_diff_pct'].values
        del bars
        gc.collect()

        new_feats, train_targs = create_hist_feats(features, targets, hist_points=hist_points, future=future)
        xform_train = scale_historical_feats_full(mkt, new_feats)
        del new_feats
        gc.collect()

        f = h5py.File(datafile, 'w')
        f.create_dataset('xform_train', data=xform_train, compression='lzf')
        f.create_dataset('train_targs', data=train_targs, compression='lzf')
        f.close()

    return xform_train, train_targs


def create_hist_feats(features, targets, dates, hist_points=480, future=24, make_all=False):
    # make historical features
    new_feats = []
    stop = features.shape[0] - future
    if make_all:
        stop += future

    for i in range(hist_points, stop):
        new_feats.append(features[i - hist_points:i, :])

    new_feats = np.array(new_feats)
    if make_all:
        new_targs = targets[hist_points + future:]
        return new_feats[:-future], targets, new_feats[-future:]
    else:
        new_targs = targets[hist_points + future:]
        dates = dates[hist_points + future:]  # dates for the targets
        return new_feats, new_targs, dates


def scale_it(dat):
    sh0, sh2 = dat.shape[0], dat.shape[2]
    s = SS(copy=False)  # copy=False does the scaling inplace, so we don't have to make a new list
    for j in tqdm(range(sh0)):  # timesteps
        for i in range(sh2):  # number of indicators/etc
            _ = s.fit_transform(dat[j, :, i].reshape(-1, 1))[:, 0]


def scale_historical_feats(feats, multiproc=False):
    # TODO: multithread so it runs faster
    if multiproc:
        cores = os.cpu_count()

        chunksize = feats.shape[0] // (cores)
        chunks = np.split(feats[:chunksize * (cores - 1), :, :], cores - 1)
        print(len(chunks))
        print(feats[np.newaxis, chunksize * (cores - 1):, :, :].shape)
        print(chunks[0].shape)
        # chunks = np.concatenate([chunks, feats[np.newaxis, chunksize * (cores - 1):, :, :]])

        pool = Pool()
        pool.map(scale_it, chunks + [feats[np.newaxis, chunksize * (cores - 1):, :, :]])
        pool.close()
        pool.join()
    else:
        scale_it(feats)


def scale_historical_feats_old(new_feats, targets, test_size=5000, test_frac=0.2):
    # also makes train/test
    # in case dataset is too small for 5k test points, adjust according to test_frac

    # TODO: multithread so it runs faster
    if new_feats.shape[0] * test_frac < test_size:
        test_size = int(new_feats.shape[0] * test_frac)

    train_feats = new_feats[:-test_size]
    train_targs = targets[:-test_size]
    test_feats = new_feats[-test_size:]
    test_targs = targets[-test_size:]

    tr_sh0, tr_sh1, tr_sh2 = train_feats.shape[0], train_feats.shape[1], train_feats.shape[2]
    te_sh0, te_sh1, te_sh2 = test_feats.shape[0], test_feats.shape[1], test_feats.shape[2]

    s = SS(copy=False)  # copy=False does the scaling inplace, so we don't have to make a new list
    for j in tqdm(range(tr_sh0)):  # timesteps
        for i in range(tr_sh2):  # number of indicators/etc
            _ = s.fit_transform(train_feats[j, :, i].reshape(-1, 1))[:, 0]

    del train_feats
    gc.collect()

    for j in tqdm(range(te_sh0)):
        for i in range(te_sh2):  # number of indicators/etc
            _ = s.fit_transform(test_feats[j, :, i].reshape(-1, 1))[:, 0]

    return train_feats, test_feats, train_targs, test_targs


def scale_historical_feats_full(mkt, train_feats):
    # dont think we need to even save the scalers...normalizing evenything regardless
    scalers = [SS(copy=False) for i in range(train_feats.shape[2])]
    for j in tqdm(range(train_feats.shape[0])):  # timesteps
        for i in range(train_feats.shape[2]):  # number of indicators/etc
            _ = scalers[i].fit_transform(train_feats[j, :, i].reshape(-1, 1))[:, 0]

    sc_file = home_dir + 'data/nn_feats_targs/poloniex/full_train_' + mkt + '_scalers.pk'
    pk.dump(scalers, open(sc_file, 'wb'))  # make sure to change this to read after copy-pasting!
    # don't need to return anything because it does it in-place


def make_latest_nn_data(mkt='BTC_AMP', points=600, future=24, hist_points=480, resamp='H'):
    """
    grabs last x points and makes data for predictions
    """
    df = pe.read_trade_hist(mkt, points=points + future)  # add 'future' points because we drop them
    # resamples to the hour
    rs_full = dp.resample_ohlc(df, resamp=resamp)
    rs_full = dp.make_mva_features(rs_full)
    bars = cts.create_tas(bars=rs_full, verbose=True)
    # make target columns
    col = str(future) + '_' + resamp + '_price_diff'
    bars[col] = bars['typical_price'].copy()
    bars[col] = np.hstack((np.repeat(bars[col].iloc[future], future), bars['typical_price'].iloc[future:].values - bars['typical_price'].iloc[:-future].values))
    bars[str(future) + '_' + resamp + '_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[future], future), bars['typical_price'].iloc[future:].values))
    # drop first 24 points because they are repeated
    bars = bars.iloc[future:]

    feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume', 'high', 'low', 'close', 'open']
    features = bars[feat_cols].values
    targets = bars[str(future) + 'h_price_diff_pct'].values

    del bars
    gc.collect()

    new_feats, train_targs = create_hist_feats(features, targets, hist_points=hist_points, future=future)
    xform_train = scale_historical_feats_full(mkt, new_feats)
    del new_feats
    gc.collect()

    return xform_train, train_targs


# TODO: plot % change with actual data to make sure they agree
import pandas as pd
mkt = 'BTC_AMP'
future = 6
hist_points = 72
df = pe.read_trade_hist(mkt)
# resamples to the hour if H, T is for minutes, S is seconds
rs_full = dp.resample_ohlc(df, resamp='H')
xform_train, xform_test, train_targs, test_targs, dates = prep_polo_nn(mkt, make_fresh=True, hist_points=hist_points, future=future, resamp='H')
all_targs = np.concatenate((train_targs, test_targs))
# drop 1000 + future in main prep_polo_nn, then get hist_points + future from create_hist_feats
rs_abbrev = rs_full.iloc[1000 + future * 2 + hist_points:]

print(rs_abbrev.index[-1])
print(pd.to_datetime(dates[-1]))

trace1 = go.Scatter(
    x=rs_abbrev.index,
    y=rs_abbrev['typical_price']
)
trace2 = go.Scatter(
    x=rs_abbrev.index,
    y=all_targs,
)

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.001)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(height=1500, width=1500)
plot(fig, filename='simple-subplot')



# make_all_nn_data(future=6, hist_points=72, resamp='H', make_fresh=True)

# 3 hours in past and 6 hours in future with 5 min bins
# xform_train, xform_test, train_targs, test_targs = prep_polo_nn('BTC_ETH', make_fresh=True, hist_points=216, future=72, resamp='5T')
# xform_train, xform_test, train_targs, test_targs = prep_polo_nn('USDT_LTC', make_fresh=True, hist_points=216, future=72, resamp='5T')
# for returning early and testing stuff
# nf, t = prep_polo_nn('USDT_LTC', make_fresh=True, hist_points=216, future=72, resamp='5T')
# xform_train, xform_test, train_targs, test_targs = prep_polo_nn('USDT_BTC', make_fresh=True, hist_points=216, future=72, resamp='5T')


# make = ['USDT_LTC', 'USDT_XRP', 'BTC_XRP', 'BTC_ETH', 'BTC_LTC']
# for m in make:
#     _, _, _, _ = prep_polo_nn(m, make_fresh=True, hist_points=72, future=6, resamp='H')

# _, _, _, _ = prep_polo_nn('BTC_DASH', make_fresh=True, hist_points=72, future=6, resamp='H')
# xform_train, xform_test, train_targs, test_targs = prep_polo_nn('USDT_BTC', make_fresh=True, hist_points=72, future=6, resamp='H')
# make_all_nn_data(future=6, hist_points=72, resamp='H', make_fresh=False, skip_load=True, save_scalers=False)
