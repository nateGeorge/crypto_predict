"""
creates, trains, loads, predicts, evaluates nueral net models
should be run from the home github directory
"""

# core
import os
import sys

# custom
sys.path.append('code/poloniex')
import prep_for_nn as pfn

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

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)


def train_net(mkt, model):
    """
    trains model on a given market
    """
    xform_train, xform_test, train_targs, test_targs = pfn.prep_polo_nn()
    mod = getattr(model)(xform_train)
    mod.fit(xform_train.reshape(xform_train.shape[0], -1), train_targs, epochs=30, validation_split=0.15, callbacks=cb, batch_size=2000)
    # next up...evaluate on train/test and save plotly plots for later inspection


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
        xform_train, xform_test, train_targs, test_targs = pfn.prep_polo_nn(mkt=p)


def stock_loss_mae_log(y_true, y_pred):
    alpha = 5.
    loss = tf.where(K.less(y_true * y_pred, 0), \
                     alpha * K.log(K.abs(y_true - y_pred) + 1), \
                     K.log(K.abs(y_true - y_pred) + 1))

    return K.mean(loss, axis=-1)


def big_dense(train_feats):
    # restart keras session (clear weights)
    K.clear_session()

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

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    cb = [es]

    mod = Model(inputs, x)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod
