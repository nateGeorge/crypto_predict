"""
creates, trains, loads, predicts, evaluates nueral net models
should be run from the home github directory
"""

# core
import os
import sys
import glob

# custom
sys.path.append('code')
sys.path.append('code/poloniex')
import prep_for_nn as pfn
from utils import get_home_dir

# installed
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import keras.backend as K
from keras.models import load_model
import keras.losses
from poloniex import Poloniex
import tensorflow as tf
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout, Candlestick

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)

HOME_DIR = get_home_dir()
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


def get_latest_model(base='5_layer_dense'):
    files = glob.glob(model_dir + base + '*')
    if len(files) != 0:
        files.sort(key=os.path.getmtime)
        return files[-1]

    return None


def train_net(mkt, model, base='5_layer_dense'):
    """
    trains model on a given market

    :param mkt: string, market pair to use for data
    :param model: function name for model to use (must exist in this file)
    :param base: base filename to search for existing models to load weights from,
                also used to save models.  model design should match exactly
    """
    xform_train, xform_test, train_targs, test_targs = pfn.prep_polo_nn(mkt=mkt)
    if xform_train is None:
        return None

    latest_mod = get_latest_model(base=base)
    if latest_mod is None:
        mod = getattr(thismodule, model)(xform_train)
    else:
        mod = getattr(thismodule, model)(xform_train)
        mod = set_weights(mod, latest_mod)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    cb = [es]

    mod.fit(xform_train.reshape(xform_train.shape[0], -1),
            train_targs,
            epochs=200,
            validation_split=0.15,
            callbacks=cb,
            batch_size=5000)
    # next up...evaluate on train/test and save plotly plots for later inspection
    mod.save(model_dir + base + '_' + mkt + '.h5')


    train_preds = mod.predict(xform_train.reshape(xform_train.shape[0], -1))[:, 0]
    train_score = mod.evaluate(xform_train.reshape(xform_train.shape[0], -1), train_targs)
    # couldn't figure this out yet
    # train_score = K.run(stock_loss_mae_log(train_targs, train_preds))
    data = [Scatter(x=train_preds, y=train_targs, mode='markers', name='preds vs actual')]
    layout = Layout(
        title='train preds vs actual (score = ' + str(train_score) + ')',
        xaxis=dict(
            title='predictions'
        ),
        yaxis=dict(
            title='actual'
        )
    )
    fig = Figure(data=data, layout=layout)
    plot(fig, filename=plot_dir + mkt + '_train_preds_vs_actual.html', auto_open=False, show_link=False)

    test_preds = mod.predict(xform_test.reshape(xform_test.shape[0], -1))[:, 0]
    test_score = mod.evaluate(xform_test.reshape(xform_test.shape[0], -1), test_targs)
    # test_score = K.run(stock_loss_mae_log(train_targs, train_preds))
    data = [Scatter(x=test_preds, y=test_targs, mode='markers', name='preds vs actual')]
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
    plot(fig, filename=plot_dir + mkt + '_test_preds_vs_actual.html', auto_open=False, show_link=False)

    return mod


def train_all_pairs():
    """
    trains all currency pairs
    """
    # gets all markets
    ticks = polo.returnTicker()
    pairs = sorted(ticks.keys())
    # for now, just BTC and USDT are maximized
    btc_pairs = [p for p in pairs if 'BTC' == p[:3]]
    usdt_pairs = [p for p in pairs if 'USDT' == p[:4]]
    pairs = btc_pairs + usdt_pairs
    for p in pairs:
        print('\n')
        print('training on ', p)
        _ = train_net(p, 'big_dense')


def train_on_pair(mkt):
    mod = train_net(mkt, 'big_dense', base='5_layer_dense')


def stock_loss_mae_log(y_true, y_pred):
    alpha1 = 10.  # penalty for predicting positive but actual is negative
    alpha2 = 2.  # penalty for predicting negative but actual is positive
    loss = tf.where(K.less(y_true * y_pred, 0), \
                     tf.where(K.less(y_true, y_pred), \
                                alpha1 * K.log(K.abs(y_true - y_pred) + 1), \
                                alpha2 * K.log(K.abs(y_true - y_pred) + 1)), \
                     K.log(K.abs(y_true - y_pred) + 1))

    return K.mean(loss, axis=-1)


def set_weights(model, old_model_file=HOME_DIR + 'notebooks/5_layer_dense_90epochs_stock_loss_mae_log_alpha5.h5'):
    old_model = load_model(old_model_file)
    for i, l in enumerate(model.layers):
        l.set_weights(old_model.layers[i].get_weights())

    return model


def big_dense(train_feats):
    """
    creates big dense model

    by default loads weights from latest trained model on BTC_STR I think
    """
    # restart keras session (clear weights)
    K.clear_session()

    # this needs to be performed before loading a model using this loss function
    keras.losses.stock_loss_mae_log = stock_loss_mae_log

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps*input_dim, ))  # timesteps, input_dim
    x = Dense(3000, activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
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
    x = Dense(1, activation='linear')(x)

    mod = Model(inputs, x)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod
