"""
creates, trains, loads, predicts, evaluates nueral net models
should be run from the home github directory
"""

# core
import os
import sys
import glob
import gc

# custom
sys.path.append('code')
sys.path.append('code/poloniex')
import prep_for_nn as pfn
import save_keras as sk
from utils import get_home_dir

# installed
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Activation, Flatten, Concatenate, Reshape
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import keras.backend as K
import keras.losses
from poloniex import Poloniex
import tensorflow as tf
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout, Candlestick
import numpy as np
import pandas as pd
import statsmodels.api as sm


key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)

HOME_DIR = get_home_dir()
HOME_DIR = '/media/nate/data_lake/crytpo_predict/'
plot_dir = HOME_DIR + 'models/poloniex/neural_net_plots/'
model_dir = HOME_DIR + 'models/poloniex/neural_nets/'

thismodule = sys.modules[__name__]

def make_dirs():
    dirs = ['models/', 'poloniex/', 'neural_nets/']
    cur_dir = HOME_DIR
    for d in dirs:
        cur_dir = cur_dir + d
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        os.mkdir(plot_dir + 'full_train')

    if not os.path.exists(model_dir + 'full_train'):
        os.mkdir(model_dir + 'full_train')


def get_latest_model(base='5_layer_dense', folder=None):
    files = glob.glob(model_dir + base + '*')
    if folder is not None:
        files = files + glob.glob(model_dir + folder + '/' + base + '*')

    if len(files) != 0:
        files.sort(key=os.path.getmtime)
        return files[-1]

    return None


def reset_collect():
    """
    clears gpu memory
    """
    tf.reset_default_graph()
    K.clear_session()
    gc.collect()


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


def remake_all_plots():
    pairs = get_btc_usdt_pairs()
    start = pairs['BTC_BTCD'].index
    for p in pairs[start:]:
        print('\n')
        print('remaking plots for', p)
        xform_train, xform_test, train_targs, test_targs = pfn.prep_polo_nn(mkt=p)
        mod = restore_model(mkt=p, base='5_layer_dense', folder='5k_test')
        if mod is None:
            continue

        plot_results(mod, p, xform_train, train_targs, xform_test, test_targs)
        del mod
        reset_collect()


def plot_results(mod, mkt, train, train_targs, test=None, test_targs=None, folder=None, test_lin_preds=None):
    # TODO: subplot with actual and predicted returns
    train_preds = mod.predict(train)[:, 0]
    train_score = mod.evaluate(train, train_targs)
    # couldn't figure this out yet
    # train_score = K.run(stock_loss_mae_log(train_targs, train_preds))
    title = 'train preds vs actual (score = ' + str(train_score) + ')'
    if test is None:
        title = 'full train preds vs actual (score = ' + str(train_score) + ')'
    data = [Scatter(x=train_preds,
                    y=train_targs,
                    mode='markers',
                    name='preds vs actual',
                    marker=dict(color=list(range(train_targs.shape[0])),
                                colorscale='Portland',
                                showscale=True,
                                opacity=0.5)
                    )]
    layout = Layout(
        title=title,
        xaxis=dict(
            title='predictions'
        ),
        yaxis=dict(
            title='actual'
        )
    )
    fig = Figure(data=data, layout=layout)
    if folder is None:
        if test is None:
            filename = plot_dir + mkt + '_full_train_preds_vs_actual.html'
        else:
            filename = plot_dir + mkt + '_train_preds_vs_actual.html'
    else:
        if not os.path.exists(plot_dir + folder):
            os.mkdir(plot_dir + folder)

        if test is None:
            filename = plot_dir + folder + '/' + mkt + '_full_train_preds_vs_actual.html'
        else:
            filename = plot_dir + folder + '/' + mkt + '_train_preds_vs_actual.html'

    plot(fig, filename=filename, auto_open=False, show_link=False)

    del train_score

    if test is not None:
        test_preds = mod.predict(test)[:, 0]
        test_score = mod.evaluate(test, test_targs)
        # test_score = K.run(stock_loss_mae_log(train_targs, train_preds))
        data = [Scatter(x=test_preds,
                        y=test_targs,
                        mode='markers',
                        name='preds vs actual',
                        marker=dict(color=list(range(test_targs.shape[0])),
                                    colorscale='Portland',
                                    showscale=True,
                                    opacity=0.5)
                        )]
        if test_lin_preds is not None:
            line = Scatter(x=test_preds,
                            y=test_lin_preds,
                            mode='lines',
                            name='linear fit')
            data = data + [line]

        layout = Layout(
            title='test preds vs actual (score = ' + str(test_score) + ')',
            xaxis=dict(
                title='predictions'
            ),
            yaxis=dict(
                title='actual'
            )
        )
        fig = Figure(data=data, layout=layout)
        if folder is None:
            filename = plot_dir + mkt + '_test_preds_vs_actual.html'
        else:
            filename = plot_dir + folder + '/' + mkt + '_test_preds_vs_actual.html'

        plot(fig, filename=filename, auto_open=False, show_link=False)

        del test_score


def get_model_path(mkt, base, folder):
    if folder is None:
        model_file = model_dir + base + '_' + mkt + '.h5'
    else:
        model_file = model_dir + folder + '/' + base + '_' + mkt + '.h5'
        if not os.path.exists(model_dir + folder):
            os.mkdir(model_dir + folder)

    return model_file


def restore_model(mkt='BTC_AMP', base='big_conv1', folder=None):
    keras.losses.stock_loss_mae_log = stock_loss_mae_log
    model_file = get_model_path(mkt, base, folder)

    if os.path.exists(model_file):
        model = sk.load_network(model_file)
        model.compile(optimizer='adam', loss=stock_loss_mae_log)
        return model

    return None


def compress_all_models():
    """
    only meant to be used once, since after this, saving/loading was changed
    to use compression
    """
    for m in sorted(glob.glob(model_dir + '*.h5')):
        print(m)
        model = load_model(m)
        sk.save_network(model, m)
        del model
        reset_collect()


def parse_deets(deets):
    # history, future, resample
    if deets is None:
        return None, None, None
    else:
        d = deets.split('_')
        return int(d[0]), int(d[1]), d[2]


def train_net(mkt, model, deets=None, base='5_layer_dense', folder=None, batch_size=2000, test=True, latest_bias=False, bias_factor=4, latest_size=3000, random_init=False):
    """
    trains model on a given market, saves 5k or 20% for testing

    :param mkt: string, market pair to use for data
    :param model: function name for model to use (must exist in this file)
    :param base: base filename to search for existing models to load weights from,
                also used to save models.  model design should match exactly
    :param folder: a folder to save models in
    :param test: if True, will save some points for testing.
    :param latest_bias: boolean, if True, will enrich latest data
                        (last number of test points in train set) by a
                        factor of bias_factor
    :param bias_factor: number of times to replicate latest test_set
                        number of points in train_set
    """
    h, f, r = parse_deets(deets)
    if test:
        xform_train, xform_test, train_targs, test_targs, dates = pfn.prep_polo_nn(mkt=mkt, hist_points=h, future=f, resamp=r)
        if xform_train is None:
            return None, None

        if xform_test is None:
            test_size = int(xform_train.shape[0] / 5)
        else:
            test_size = xform_test.shape
    else:
        xform_train, train_targs, dates = pfn.make_polo_nn_fulltrain(mkt=mkt, hist_points=h, future=f, resamp=r)
        xform_test, test_targs = None, None

    if xform_train is None:
        return None, None

    if latest_size > int(train_targs.shape[0] / 4):
        latest_size = int(train_targs.shape[0] / 4)

    test_size = latest_size

    val_frac = 0.15
    if latest_bias:
        val_size = int(val_frac * train_targs.shape[0])
        train_eval = np.copy(xform_train)  # save original for evaluation
        train_eval_targs = np.copy(train_targs)
        start = -(test_size + val_size)
        for i in range(bias_factor):
            xform_train = np.vstack((xform_train, xform_train[start:-val_size, :, :]))
            train_targs = np.hstack((train_targs, train_targs[start:-val_size]))

    if random_init:
        latest_mod = None
    else:
        latest_mod = get_latest_model(base=base, folder=folder)

    if latest_mod is None:
        mod = getattr(thismodule, model)(xform_train)
    else:
        mod = getattr(thismodule, model)(xform_train)
        mod = set_weights(mod, latest_mod)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=0, mode='auto')
    cb = [es]

    if model in ['big_dense']:
        train = xform_train.reshape(xform_train.shape[0], -1)
        if latest_bias:
            train_eval = train_eval.reshape(train_eval.shape[0], -1)
        test = xform_test.reshape(xform_test.shape[0], -1)
    elif model in ['big_conv1', 'conv1d_lstm', 'big_lstm_conv', 'smaller_conv1', 'conv1d_lstm_small']:
        train = xform_train
        test = xform_test

    mem_use = get_model_memory_usage(batch_size, mod)
    print('expected memory usage:', mem_use)

    history = mod.fit(train,
                    train_targs,
                    epochs=200,
                    validation_split=val_frac,
                    callbacks=cb,
                    batch_size=batch_size)

    model_file = get_model_path(mkt, base, folder)
    print('saving as', model_file)
    sk.save_network(mod, model_file)

    best, lin_preds = get_best_thresh_linear_model(mod, test, test_targs)

    if latest_bias:
        plot_results(mod, mkt, train_eval, train_eval_targs, test, test_targs, folder=folder, test_lin_preds=lin_preds)
        best_99pct = get_best_thresh(mod, train_eval, train_eval_targs, cln=False)
    else:
        plot_results(mod, mkt, train, train_targs, test, test_targs, folder=folder, test_lin_preds=lin_preds)
        best_99pct = get_best_thresh(mod, train=train, train_targs=train_targs, test=test, test_targs=test_targs, cln=False)


    del history
    gc.collect()

    return mod, best


def train_all_pairs(test=True, latest_bias=True, model='smaller_conv1', deets='72_6_H', folder='smaller_conv1', batch_size=2000, random_init=False):
    """
    trains all currency pairs
    :param test: if True, will keep a test set.  otherwise combines train and test and
                trains on all
    """
    # added folder arg, so don't want this for now
    # folder = None
    # if not test:
    #     folder = 'full_train'

    pairs = get_btc_usdt_pairs()
    best_threshes = {'market': [], 'threshold': []}
    pairs = get_btc_usdt_pairs()
    # added to start with random weights on first one
    start = pairs.index('BTC_BCH')
    for p in pairs[start:]:
        print('\n'*3)
        print('training on', p)
        mod, best = train_net(p,
                            model=model,  # model function name
                            deets=deets,
                            batch_size=batch_size,
                            test=test,
                            latest_bias=latest_bias,
                            folder=folder,
                            random_init=random_init)
        if mod is None:
            continue

        best_threshes['market'].append(p)
        best_threshes['threshold'].append(best)
        del mod
        reset_collect()

    df = pd.DataFrcame(best_threshes)
    df.set_index('market', inplace=True)
    if test:
        thresh_file = model_dir + 'best_threshes_with_test.h5'
    else:
        thresh_file = model_dir + 'best_threshes_no_test.h5'

    df.to_hdf(thresh_file, 'data', mode='w', complib='blosc', complevel=9)


def train_on_one(mkt, model, deets=None, test=True, latest_bias=True, folder=None, random_init=False, batch_size=1000):
    mod, best = train_net(mkt,
                        model,
                        deets=deets,
                        base=model,
                        batch_size=batch_size,
                        test=test,
                        latest_bias=latest_bias,
                        folder=folder,
                        random_init=random_init)

    return mod, best


def stock_loss_mae_log(y_true, y_pred):
    # was using 8 and 2, but seemed to bias models towards just predicting everything negative
    # don't want the penalties too big, or it will just fit everything to a small number
    alpha1 = 1.  # penalty for predicting positive but actual is negative
    alpha2 = 1.  # penalty for predicting negative but actual is positive
    loss = tf.where(K.less(y_true * y_pred, 0), \
                     tf.where(K.less(y_true, y_pred), \
                                alpha1 * K.log(K.abs(y_true - y_pred) + 1), \
                                alpha2 * K.log(K.abs(y_true - y_pred) + 1)), \
                     K.log(K.abs(y_true - y_pred) + 1))

    return K.mean(loss, axis=-1)


def stock_loss_mae_log_pct(y_true, y_pred):
    # doesn't seem to work well
    # was using 8 and 2, but seemed to bias models towards just predicting everything negative
    # don't want the penalties too big, or it will just fit everything to a small number
    alpha1 = 1.  # penalty for predicting positive but actual is negative
    alpha2 = 1.  # penalty for predicting negative but actual is positive
    loss = tf.where(K.less(y_true * y_pred, 0), \
                     tf.where(K.less(y_true, y_pred), \
                                alpha1 * K.log(K.abs((y_true - y_pred)/K.clip(y_true, K.epsilon(), None)) + 1), \
                                alpha2 * K.log(K.abs((y_true - y_pred)/K.clip(y_true, K.epsilon(), None)) + 1)), \
                     K.log(K.abs((y_true - y_pred)/K.clip(y_true, K.epsilon(), None)) + 1))

    return K.mean(loss, axis=-1)


# this needs to be performed before loading a model using this loss function
keras.losses.stock_loss_mae_log = stock_loss_mae_log
keras.losses.stock_loss_mae_log_pct = stock_loss_mae_log_pct


def set_weights(model, old_model_file):
    old_model = sk.load_network(old_model_file)
    for i, l in enumerate(model.layers):
        l.set_weights(old_model.layers[i].get_weights())

    del old_model

    return model


def big_dense(train_feats):
    """
    creates big dense model

    need to reshape input data like:
        train_feats.reshape(train_feats.shape[0], -1)

    by default loads weights from latest trained model on BTC_STR I think
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps*input_dim, ))
    x = Dense(3000, activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='linear')(x)

    mod = Model(inputs, x)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def big_conv1(train_feats):
    """
    creates big convolutional model
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def smaller_conv1(train_feats):
    """
    creates big convolutional model
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def conv1d_lstm(train_feats):
    K.clear_session()
    tf.reset_default_graph()

    # inputs used for both halves
    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))

    # first half of net - purely convolutional
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)
    x = Flatten()(x)

    # second half of net - lstm
    l = LSTM(1000, dropout=0.4, return_sequences=True, activation='tanh')(inputs)
    l = BatchNormalization()(l)
    l = LSTM(500, activation='tanh', dropout=0.4)(l)
    l = BatchNormalization()(l)
    l = Dense(1000)(l)

    # combine them
    x = Concatenate()([x, l])
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    adam = optimizers.Adam(lr=0.001)
    mod.compile(optimizer=adam, loss=stock_loss_mae_log)

    return mod


def conv1d_lstm_small(train_feats):
    K.clear_session()
    tf.reset_default_graph()

    # inputs used for both halves
    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))

    # first half of net - purely convolutional
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)
    x = Flatten()(x)

    # second half of net - lstm
    l = LSTM(1000, dropout=0.4, return_sequences=True, activation='tanh')(inputs)
    l = BatchNormalization()(l)
    l = LSTM(500, activation='tanh', dropout=0.4)(l)
    l = BatchNormalization()(l)
    l = Dense(1000)(l)

    # combine them
    x = Concatenate()([x, l])
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    adam = optimizers.Adam(lr=0.001)
    mod.compile(optimizer=adam, loss=stock_loss_mae_log)

    return mod


def big_lstm_conv(train_feats):
    # restart keras session (clear weights)
    K.clear_session()

    timesteps = train_feats.shape[1]
    lstm_dim = 1000
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))  # timesteps, input_dim
    x = LSTM(lstm_dim, dropout=0.4, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(lstm_dim)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Reshape((-1, 1))(x)
    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    adam = optimizers.Adam(lr=0.001)
    mod.compile(optimizer=adam, loss=stock_loss_mae_log)

    return mod


def round_to_005(x):
    """
    rounds to nearest 0.005 and makes it pretty (avoids floating point 0.000001 nonsense)
    """
    res = round(x * 200) / 200
    return float('%.3f'%res)


def get_best_thresh_linear_model(mod, test, test_targs):
    preds = mod.predict(test)[:, 0]
    X = sm.add_constant(preds, has_constant='add')
    y = test_targs
    lin_model = sm.OLS(y, X).fit()
    x_line = np.arange(min(preds), max(preds), 100)
    intercept, slope = lin_model.params  # intercept, slope
    # to get the value of x at y=0 (the x-intercept, where on average, the actual returns are positive)
    # take -b/m, or -intercetp/slope
    xint = -intercept / slope
    # add 10% for safety margin
    best = 1.1 * xint

    return best, lin_model.fittedvalues


def get_best_thresh(mod, train, train_targs, test=None, test_targs=None, cln=True, verbose=False):
    # want to use both train and test sets, but want to plot each separately
    if test is not None:
        train = np.vstack((train, test))
        train_targs = np.hstack((train_targs, test_targs))
        xform_test, test_targs = None, None

    train_preds = mod.predict(train)[:, 0]

    if cln:
        del mod
        reset_collect()

    # bin into 0.5% increments for the predictions
    max_pred = max(train_preds)
    best = max_pred
    hi = round_to_005(max_pred) # rounds to nearest half pct
    lo = round_to_005(hi - 0.005)  # 0.5% increments
    cumulative_agree = 0
    cumulative_pts = 0
    while lo > 0:
        mask = (train_preds >= lo) & (train_preds < hi)
        lo -= 0.005
        hi -= 0.005
        lo = round_to_005(lo)
        hi = round_to_005(hi)
        filtered = train_targs[mask]
        if filtered.shape[0] == 0:
            continue

        pos = filtered[filtered >= 0].shape[0]
        cumulative_agree += pos
        cumulative_pts += filtered.shape[0]
        pct_agree = pos/filtered.shape[0]
        if pct_agree >= 0.99 and cumulative_agree >= 0.99:
            best = lo

        if verbose:
            print('interval:', '%.3f'%lo, '%.3f'%hi, ', pct agreement:', '%.3f'%pct_agree)

    return best


def find_best_thresh(mkt, make_plots=False, verbose=False):
    """
    finds best threshold where predictions are positive and highly accurate
    """
    xform_train, xform_test, train_targs, test_targs = pfn.prep_polo_nn(mkt=mkt)
    if xform_train is None:
        return None

    mod = restore_model(mkt=mkt, base='5_layer_dense')
    if mod is None:
        return None

    if make_plots:
        plot_results(mod, mkt, xform_train, train_targs, xform_test, test_targs)

    best = get_best_thresh(mod, xform_train, train_targs, xform_test, test_targs, cln=True, verbose=verbose)

    return best


def find_all_best(make_plots=False):
    """
    finds best thresholds for 99% same direction accuracy.
    can also re-make plots

    :param make_plots: if true, will make plots too
    """
    pairs = get_btc_usdt_pairs()
    best_threshes = {'market': [], 'threshold': []}
    for p in pairs:
        print('\nfinding best thresh for', p)
        best = find_best_thresh(p, make_plots=make_plots)
        if best is None:
            continue

        best_threshes['market'].append(p)
        best_threshes['threshold'].append(best)

    return best_threshes


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


if __name__ == "__main__":
    """
    hourly bins
        on BTC_AMP with 480 history pts and predicting 6 hours into the future, didn't do great on test, but starting to show
    some of the correct structure with conv1d_lstm
    24 into the future was total junk with the 5 layer dense

    72 hist, 6 future on dash with big_dense on BTC_DASH:
        train/test look ok, losses of 0.03/0.05
    with smaller_conv1:
        looking decent!  but actual/pred plots had weird shape.  probably need to train more?

    conv1d_lstm: looks like it's getting close, need more training?

    5-minutely bins
    18 hours in past and 6 in future on BTC_ETH with big_dense:


    # 11-6-2017: realized a bug in the data creation code...was dividing returns by the current date, not past, so they were incorrect
    # also was creating prediction data incorrectly; fixed it but now predictions not working well

    smaller_conv1 seems to be working decently, but not on unseen data.  with larger datasets seems to work best
    """
    # run after data has been created
    #train_all_pairs()


    # mod, best = train_on_one('BTC_DASH', 'big_dense', random_init=False, batch_size=2000)
    # mod, best = train_on_one('BTC_DASH', model='big_dense', deets='72_6_H', folder=None, random_init=False, batch_size=2000)
    # del mod
    # mod, best = train_on_one('BTC_DASH', model='smaller_conv1', deets='72_6_H', folder='smaller_conv1', random_init=False, batch_size=2000)
    # del mod
    mod, best = train_on_one('USDT_BTC', model='smaller_conv1', deets='72_6_H', folder='smaller_conv1', random_init=False, batch_size=2000)
    del mod
    mod, best = train_on_one('USDT_LTC', model='smaller_conv1', deets='72_6_H', folder='smaller_conv1', random_init=False, batch_size=2000)
    del mod
    mod, best = train_on_one('BTC_DASH', model='conv1d_lstm_small', deets='72_6_H', folder='conv1d_lstm_small', random_init=True, batch_size=25)
    del mod
    mod, best = train_on_one('USDT_BTC', model='conv1d_lstm_small', deets='72_6_H', folder='conv1d_lstm_small', random_init=True, batch_size=25)
    del mod
    mod, best = train_on_one('USDT_LTC', model='conv1d_lstm_small', deets='72_6_H', folder='conv1d_lstm_small', random_init=True, batch_size=25)
    del mod
    #
    #
    # # mod, best = train_on_one('BTC_AMP', 'conv1d_lstm', folder='conv1d_lstm', random_init=False, batch_size=100)
    # # pass
    # # mod, best = train_on_one('BTC_AMP', 'conv1d_lstm', folder='conv1d_lstm', random_init=False, batch_size=100)  # 6 h in future right now
    # # del mod
    # mod, best = train_on_one('BTC_ETH', model='conv1d_lstm', deets='216_72_5T', folder='conv1d_lstm', random_init=True, batch_size=100)  # still 24h in future for next 3
    # del mod
    # mod, best = train_on_one('USDT_LTC', model='conv1d_lstm', deets='216_72_5T', folder='conv1d_lstm', random_init=True, batch_size=100, latest_bias=False)  # still 24h in future for next 3
    # del mod
    # gc.collect()
    # mod, best = train_on_one('USDT_BTC', model='conv1d_lstm', deets='216_72_5T', folder='conv1d_lstm', random_init=True, batch_size=100, latest_bias=False)  # still 24h in future for next 3
    # del mod


    # del mod
    # mod, best = train_on_one('BTC_STR', 'conv1d_lstm', folder='conv1d_lstm', random_init=False, batch_size=100)
    # del mod
    # mod, best = train_on_one('BTC_DASH', 'conv1d_lstm', folder='conv1d_lstm', random_init=False, batch_size=100)
    # del mod
    #
    # mod, best = train_on_one('BTC_AMP', 'big_lstm_conv', folder='big_lstm_conv', random_init=True, batch_size=25)
    # del mod
    # mod, best = train_on_one('BTC_ETH', 'big_lstm_conv', folder='big_lstm_conv', random_init=False, batch_size=25)
    # del mod
    # mod, best = train_on_one('BTC_DASH', 'big_lstm_conv', folder='big_lstm_conv', random_init=False, batch_size=25)
    # del mod
    # train_all_pairs(test=True, latest_bias=True)
    # train_all_pairs(test=False, latest_bias=True)
    #
    # train_all_pairs(test=False, latest_bias=True)
    #
    # best_threshes = find_all_best(make_plots=True)
    # df = pd.DataFrame(best_threshes)
    # df.set_index('market', inplace=True)
    # thresh_file = model_dir + 'best_threshes_with_test.h5'
    # df.to_hdf(thresh_file, 'data', mode='w', complib='blosc', complevel=9)
