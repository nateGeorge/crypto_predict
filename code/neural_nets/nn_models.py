# core
import sys
import time

# internal
sys.path.append('../poloniex')
sys.path.append('..')
import polo_eda as pe
import calc_TA_sigs as cts
import data_processing as dp

# installed
import tensorflow as tf
import pandas as pd
import numpy as np

# plotting
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import Scatter, Figure, Layout, Candlestick
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go


def data_input(feats):
    """
    Return a Tensor for a batch of data input
    :param feat: a single feature from the features array
    :returns: Tensor placeholder for image input.
    """
    return tf.placeholder(tf.float32, [None, *feats.shape[1:]], name='input')


def data_target():
    """
    Return a Tensor for output
    :returns: Tensor for label input.
    """
    # DONE: Implement Function
    return tf.placeholder(tf.float32, shape=(None, 1), name='target')


def keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # DONE: Implement Function
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return keep_prob


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs)


def make_batches(feats, targs, batchsize=32):
    f_batches = []
    t_batches = []
    num_batches = int(targs.shape[0] / batchsize)
    print('num_batches:', num_batches)
    print('leftovers:', targs.shape[0] - batchsize*num_batches)
    for i in range(num_batches):
        start_idx = i * batchsize
        end_idx = start_idx + batchsize
        f_batches.append(feats[start_idx:end_idx])
        t_batches.append(targs[start_idx:end_idx])

    print(i)
    f_batches.append(feats[end_idx:])
    t_batches.append(targs[end_idx:])
    f_batches = np.array(f_batches)
    t_batches = np.array(t_batches)

    return f_batches, t_batches


def simple_net(x, keep_prob):
    flat = tf.contrib.layers.flatten(x)
    fc1 = fully_conn(flat, 500)
    drop1 = tf.nn.dropout(fc1, keep_prob)
    out = tf.layers.dense(drop1, 1)  # need a linear activation on the output to work
    return out


def train_neural_network(session, feed_dict, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # DONE: Implement Function
    # tf.initialize_all_variables()
    session.run(optimizer, feed_dict=feed_dict)


def print_stats(session, feed_dict, mse):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = session.run(mse, feed_dict=feed_dict)
    # val_loss = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})

    print('loss = {0}'.format(loss))


def simple_nn_model_future(market='BTC_AMP',
                            datapoints=10000,
                            history=300,
                            future=180):
    """
    Predicts out to furthest future point
    """
    pass


def simple_nn_model_prototype(market='BTC_AMP',
                            datapoints=10000,
                            teston=1000,
                            history=300,
                            future=180):
    """
    prototype of simple neural net model
    """
    # Remove previous weights, bias, inputs, etc...just in case
    tf.reset_default_graph()

    df = pe.read_trade_hist(market)
    rs = dp.resample_ohlc(df)
    sc_df, scalers = dp.transform_data(rs)
    feats, targs = dp.create_hist_feats(sc_df.iloc[-datapoints:],
                                        history=300,
                                        future=future)
    train_feats = feats[:-teston]
    train_targs = targs[:-teston]

    # hyperparameters
    epochs = 15
    batch_size = 32
    keep_probability = 0.5

    # Inputs
    x = data_input(feats)
    y = data_target()
    keep_prob = keep_prob_input()

    # Model
    pred = simple_net(x, keep_prob)

    # Name predictions Tensor, so that is can be loaded from disk after training
    pred = tf.identity(pred, name='predictions')

    # Loss and Optimizer
    mse = tf.losses.mean_squared_error(pred, y)
    loss = tf.reduce_mean(mse, name='mse_mean')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    save_model_path = './simple_model'
    sess = tf.Session()

    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    # IMPORTANT! only train on 90% of data
    f_batches, t_batches = make_batches(train_feats, train_targs, batch_size)
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        for feature_batch, label_batch in zip(f_batches, t_batches):
            # train the net
            fd = fd = {x: feature_batch,
                        y: label_batch,
                        keep_prob: keep_probability}
            sess.run(optimizer, feed_dict=fd)

        print('Epoch {:>2}:  '.format(epoch + 1), end='')
        fd = {x: feature_batch, y: label_batch, keep_prob: 1.}
        print_stats(sess, fd, mse)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

    preds = sess.run(pred, feed_dict={x: feats, y: targs, keep_prob: 1})
    targ_df = pd.DataFrame({'close': targs.flatten()})
    pred_df = pd.DataFrame({'close': preds.flatten()})
    rs_idx = history + future - datapoints
    resc_targs = dp.reform_data(rs[rs_idx:], targ_df, scalers)
    resc_preds = dp.reform_data(rs[rs_idx:], pred_df, scalers)
    data = [go.Scatter(x=rs[rs_idx:].index, y=resc_targs.close, name='actual'),
           go.Scatter(x=rs[rs_idx:].index, y=resc_preds.close, name='predictions')]
    plot(data)


def simple_nn_model_score(market='BTC_AMP',
                            datapoints=10000,
                            teston=1000,
                            history=300,
                            future=180):
    """
    trains model on datapoints-teston, then predicts points 'future' in the future,
    then steps forward one point and does it until all teston-future points have
    been tested, then scores and plots it
    """
    # Remove previous weights, bias, inputs, etc...just in case
    tf.reset_default_graph()

    df = pe.read_trade_hist(market)
    rs = dp.resample_ohlc(df)
    sc_df, scalers = dp.transform_data(rs)
    feats, targs = dp.create_hist_feats(sc_df.iloc[-datapoints:],
                                        history=300,
                                        future=future)
    train_feats = feats[:-teston]
    train_targs = targs[:-teston]

    # hyperparameters
    epochs = 15
    batch_size = 32
    keep_probability = 0.5

    # Inputs
    x = data_input(feats)
    y = data_target()
    keep_prob = keep_prob_input()

    # Model
    pred = simple_net(x, keep_prob)

    # Name predictions Tensor, so that is can be loaded from disk after training
    pred = tf.identity(pred, name='predictions')

    # Loss and Optimizer
    mse = tf.losses.mean_squared_error(pred, y)
    loss = tf.reduce_mean(mse, name='mse_mean')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    save_model_path = './simple_model'
    sess = tf.Session()

    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    # IMPORTANT! only train on 90% of data
    f_batches, t_batches = make_batches(train_feats, train_targs, batch_size)
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        for feature_batch, label_batch in zip(f_batches, t_batches):
            # train the net
            fd = fd = {x: feature_batch,
                        y: label_batch,
                        keep_prob: keep_probability}
            sess.run(optimizer, feed_dict=fd)

        print('Epoch {:>2}:  '.format(epoch + 1), end='')
        fd = {x: feature_batch, y: label_batch, keep_prob: 1.}
        print_stats(sess, fd, mse)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

    preds = sess.run(pred, feed_dict={x: feats, y: targs, keep_prob: 1})
    targ_df = pd.DataFrame({'close': targs.flatten()})
    pred_df = pd.DataFrame({'close': preds.flatten()})
    rs_idx = history + future - datapoints
    resc_targs = dp.reform_data(rs[rs_idx:], targ_df, scalers)
    resc_preds = dp.reform_data(rs[rs_idx:], pred_df, scalers)
    data = [go.Scatter(x=rs[rs_idx:].index, y=resc_targs.close, name='actual'),
           go.Scatter(x=rs[rs_idx:].index, y=resc_preds.close, name='predictions')]
    plot(data)


if __name__=="__main__":
    simple_nn_model_prototype()
