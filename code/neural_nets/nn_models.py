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
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
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


class nn_model:
    def __init__(self,
                market='BTC_AMP',
                future=180,
                history=300,
                mva=30,
                train_pts=10000,
                test_pts=1000,
                model='simple_fc'):
        self.market = market
        self.future = future
        self.history = history
        self.mva = mva
        self.train_pts = train_pts
        self.test_pts = test_pts
        self.model = model


    def create_data(self, market=None):
        if market is not None:
            self.market = market

        self.df = pe.read_trade_hist(self.market)
        self.rs = dp.resample_ohlc(self.df)
        self.sc_df, self.scalers = dp.transform_data(self.rs, mva=self.mva)
        self.feats, self.targs = dp.create_hist_feats(self.sc_df.iloc[-self.train_pts:],
                                                    history=self.history,
                                                    future=self.future)
        self.train_feats = self.feats[:-self.test_pts]
        self.train_targs = self.targs[:-self.test_pts]


    def data_input(self, feats):
        """
        Return a Tensor for a batch of data input
        :param feat: a single feature from the features array
        :returns: Tensor placeholder for image input.
        """
        return tf.placeholder(tf.float32, [None, *feats.shape[1:]], name='input')


    def data_target(self):
        """
        Return a Tensor for output
        :returns: Tensor for label input.
        """
        # DONE: Implement Function
        return tf.placeholder(tf.float32, shape=(None, 1), name='target')


    def keep_prob_input(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """
        # DONE: Implement Function
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return keep_prob


    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        return tf.contrib.layers.fully_connected(x_tensor, num_outputs)


    def make_batches(self, feats, targs, batchsize=32):
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


    def print_stats(self, session, feed_dict, mse):
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


    def simple_fc(self, x, keep_prob):
        flat = tf.contrib.layers.flatten(x)
        fc1 = self.fully_conn(flat, 500)
        drop1 = tf.nn.dropout(fc1, keep_prob)
        out = tf.layers.dense(drop1, 1)  # need a linear activation on the output to work
        return out


    def set_hyperparameters(self, epochs=15, batch_size=32, keep_prob=0.5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_probability = keep_prob


    def create_graph(self):
        # first reset to avoid any errors from any graphs currently loaded
        tf.reset_default_graph()
        # Inputs
        self.x = self.data_input(self.feats)
        self.y = self.data_target()
        # confusingly, keep_prob is the placeholder, and keep_probability is
        # the float that holds the value
        self.keep_prob = self.keep_prob_input()

        # Model
        # self.pred = simple_net(x, keep_prob)  # old way of doing it
        self.pred = getattr(self, self.model)(self.x, self.keep_prob)

        # Name predictions Tensor, so that is can be loaded from disk after training
        self.pred = tf.identity(self.pred, name='predictions')

        # Loss and Optimizer
        self.mse = tf.losses.mean_squared_error(self.pred, self.y)
        self.loss = tf.reduce_mean(self.mse, name='mse_mean')
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.save_model_path = './' + self.model
        self.sess = tf.Session()

        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())


    def train_net(self):
        # IMPORTANT! only train on 90% of data
        f_batches, t_batches = self.make_batches(self.train_feats,
                                                self.train_targs,
                                                self.batch_size)
        # Training cycle
        for epoch in range(self.epochs):
            # Loop over all batches
            for feature_batch, label_batch in zip(f_batches, t_batches):
                # train the net
                fd = {self.x: feature_batch,
                        self.y: label_batch,
                        self.keep_prob: self.keep_probability}
                self.sess.run(self.optimizer, feed_dict=fd)

            print('Epoch {:>2}:  '.format(epoch + 1), end='')
            fd = {self.x: feature_batch, self.y: label_batch, self.keep_prob: 1.}
            self.print_stats(self.sess, fd, self.mse)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_model_path)


    def get_all_preds(self, plot_data=True, nb=False):
        fd = {self.x: self.feats, self.y: self.targs, self.keep_prob: 1.0}
        all_preds = self.sess.run(self.pred, feed_dict=fd)
        targ_df = pd.DataFrame({'close': self.targs.flatten()})
        pred_df = pd.DataFrame({'close': all_preds.flatten()})
        rs_idx = self.history + self.future - self.train_pts
        resc_targs = dp.reform_data(self.rs[rs_idx:], targ_df, self.scalers, mva=self.mva)
        resc_preds = dp.reform_data(self.rs[rs_idx:], pred_df, self.scalers, mva=self.mva)
        if plot_data:
            data = [go.Scatter(x=self.rs[rs_idx:].index, y=resc_targs.close, name='actual'),
                   go.Scatter(x=self.rs[rs_idx:].index, y=resc_preds.close, name='predictions')]
            iplot(data) if nb else plot(data)


    def pred_future_points_all(sc_df,
                           teston=30,
                           history=300,
                           future=180,
                           mva=30):
        """
        Predicts the future points one at a time in the 'teston' section.
        teston must be at least the size of the mva used to normalize, otherwise ref_targs comes back all na's
        """
        last_preds = None
        start = -(teston + history + future)
        feats, targs = dp.create_feats_to_current(sc_df.iloc[start:], history=history, future=future)
        targ_df = pd.DataFrame({'close': targs.flatten()})
        ref_targs = dp.reform_data(rs[-(teston + mva):], targ_df, scalers)
        for i in range(teston):
            preds = sess.run(pred, feed_dict={x: feats[i:future + i - teston], y: targs[i].reshape(-1, 1), keep_prob: 1})
            fut_idx = future - 1 - teston
            future_preds = preds[fut_idx:]
            resc_preds = dp.rescale_data(pd.DataFrame({'close': future_preds.flatten()}), scalers)
            ref_future_pred = dp.reform_prediction(rs, resc_preds.close.values)
            ref_future_pred = pd.DataFrame({'close': ref_future_pred})

            last_pred = ref_future_pred.iloc[-1:].close
            if last_preds is None:
                last_preds = last_pred
            else:
                last_preds = last_preds.append(last_pred)

        future_idx = sc_df.index[-teston:]
        last_preds = pd.DataFrame(last_preds)
        last_preds.set_index(future_idx, inplace=True)

        return ref_targs, last_preds

    rft, rfp = pred_future_points_all(sc_df)

if __name__=="__main__":
    # simple_nn_model_prototype()
    nn = nn_model()
    nn.create_data()
    nn.create_graph()
    nn.set_hyperparameters()
    nn.train_net()
    nn.get_all_preds()
