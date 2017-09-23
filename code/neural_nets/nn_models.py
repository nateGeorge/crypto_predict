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
                mva=None,
                train_pts=10000,
                test_pts=1000,
                model='simple_8layer_fc'):
        self.market = market
        self.future = future
        self.history = history
        # a mva of 'None' will not use mva scaling.  Still having trouble getting
        # predictions to look good with mva scaling
        self.mva = mva
        self.train_pts = train_pts
        self.test_pts = test_pts
        self.model = model
        self.future_targs = None
        self.future_preds = None
        self.past_preds = None
        self.past_targs = None
        self.future_scores = [] # list for holding mse of future predictions
        self.past_scores = []
        self.rs = None


    def create_data(self, market=None):
        if market is not None:
            self.market = market

        self.df = pe.read_trade_hist(self.market)
        self.rs = dp.resample_ohlc(self.df)
        self.sc_df, self.scalers = dp.transform_data(self.rs.iloc[-self.train_pts:],
                                                    mva=self.mva)
        self.feats, self.targs = dp.create_hist_feats(self.sc_df,
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


    def learning_rate_input(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """
        # DONE: Implement Function
        lr = tf.placeholder(tf.float32, name='learning_rate')
        return lr


    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        return tf.contrib.layers.fully_connected(x_tensor,
                                                num_outputs,
                                                activation_fn=tf.nn.elu)


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


    def print_stats(self, session, feed_dict):
        """
        Print information about loss and validation accuracy
        : session: Current TensorFlow session
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """
        loss = session.run(self.loss, feed_dict=feed_dict)
        # val_loss = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})

        print('loss = {0}'.format(loss))


    def simple_fc(self, x, keep_prob):
        flat = tf.contrib.layers.flatten(x)
        fc1 = self.fully_conn(flat, 500)
        drop1 = tf.nn.dropout(fc1, keep_prob)
        out = tf.layers.dense(drop1, 1)  # need a linear activation on the output to work
        return out


    def simple_3layer_fc(self, x, keep_prob):
        flat = tf.contrib.layers.flatten(x)
        fc1 = self.fully_conn(flat, 500)
        drop1 = tf.nn.dropout(fc1, keep_prob)
        fc2 = self.fully_conn(drop1, 800)
        drop2 = tf.nn.dropout(fc2, keep_prob)
        fc3 = self.fully_conn(drop2, 300)
        drop3 = tf.nn.dropout(fc3, keep_prob)
        out = tf.layers.dense(drop3, 1)  # need a linear activation on the output to work
        return out


    def simple_8layer_fc(self, x, keep_prob):
        flat = tf.contrib.layers.flatten(x)
        fc1 = self.fully_conn(flat, 300)
        drop1 = tf.nn.dropout(fc1, keep_prob)
        fc2 = self.fully_conn(drop1, 400)
        fc3 = self.fully_conn(fc2, 500)
        drop3 = tf.nn.dropout(fc3, keep_prob)
        fc4 = self.fully_conn(drop3, 500)
        fc5 = self.fully_conn(fc4, 400)
        drop5 = tf.nn.dropout(fc5, keep_prob)
        fc6 = self.fully_conn(drop5, 300)
        fc7 = self.fully_conn(fc6, 200)
        drop7 = tf.nn.dropout(fc7, keep_prob)
        fc8 = self.fully_conn(drop7, 100)
        out = tf.layers.dense(fc8, 1)  # need a linear activation on the output to work
        return out


    def simple_lstm(self, x, keep_prob, lstm_size=300):
        tf.contrib.rnn.BasicLSTMCell(lstm_size)


    def set_hyperparameters(self, epochs=50, batch_size=32, keep_prob=0.5, lr=0.001):
        """
        must be called before create_graph()
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_probability = keep_prob
        self.lr = lr


    def create_graph(self):
        # first reset to avoid any errors from any graphs currently loaded
        tf.reset_default_graph()
        # Inputs
        self.x = self.data_input(self.feats)
        self.y = self.data_target()

        # confusingly, keep_prob is the placeholder, and keep_probability is
        # the float that holds the value
        self.keep_prob = self.keep_prob_input()
        self.learning_rate = self.learning_rate_input()

        # Model
        # self.pred = simple_net(x, keep_prob)  # old way of doing it
        self.pred = getattr(self, self.model)(self.x, self.keep_prob)

        # Name predictions Tensor, so that is can be loaded from disk after training
        self.pred = tf.identity(self.pred, name='predictions')

        # Loss and Optimizer
        self.mse = tf.losses.mean_squared_error(self.pred, self.y)
        self.loss = tf.reduce_mean(self.mse, name='mse_mean')
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('mse', self.mse)
        # can also minimize loss, but that number is typically way smaller than
        # mse because it's averaged

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

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
            self.print_stats(self.sess, fd)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_model_path)


    def get_all_preds(self, plot_data=False, nb=False):
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

        self.all_past_preds = resc_preds
        self.all_past_targs = resc_targs

        return resc_targs, resc_preds


    def get_past_preds(self, past=None):
        """
        past can be used to limit the amount of points plotted
        need to double-check the math with the 'past' indexing
        need to double-check the 'start' variable
        """
        if past is None:
            self.past = self.train_pts - self.test_pts
            feats = self.train_feats
            targs = self.train_targs
            start = - self.train_pts + self.future + self.history
        else:
            self.past = past
            start = -(self.past + self.test_pts)
            feats, targs = dp.create_hist_feats(self.sc_df.iloc[start:-self.test_pts],
                                                history=self.history,
                                                future=self.future)

        fd = {self.x: feats, self.y: targs, self.keep_prob: 1}
        preds = self.sess.run(self.pred, feed_dict=fd)
        targ_df = pd.DataFrame({'close': targs.flatten()})
        pred_df = pd.DataFrame({'close': preds.flatten()})
        resc_preds = dp.reform_data(self.rs[start:-self.test_pts],
                                    pred_df,
                                    self.scalers,
                                    mva=self.mva)
        resc_targs = dp.reform_data(self.rs[start:-self.test_pts],
                                    targ_df,
                                    self.scalers,
                                    mva=self.mva)

        if self.mva is None:
            # if past is not None, probably need to change this indexing...
            idx = self.rs[start:-self.test_pts].index
            resc_targs.set_index(idx, inplace=True)
            resc_preds.set_index(idx, inplace=True)

        self.past_targs = resc_targs
        self.past_preds = resc_preds

        return resc_targs, resc_preds


    def pred_future_points_cheating(self):
        """
        Predicts the future points one at a time in the self.test_pts section.
        test_pts must be at least the size of the mva used to normalize, otherwise future_targs comes back all na's

        This version uses the actual data up to the point before the prediction,
        when actually there won't be data starting at the first future point.
        So the predictions look better than they actually are.
        """
        last_preds = None
        start = -(self.test_pts + self.history + self.future)
        rs_start = -(self.test_pts + self.mva - 1)
        feats, targs = dp.create_feats_to_current(self.sc_df.iloc[start:],
                                                    history=self.history,
                                                    future=self.future)
        targ_df = pd.DataFrame({'close': targs.flatten()})
        future_targs = dp.reform_data(self.rs[-(self.test_pts + self.mva):],
                                    targ_df,
                                    self.scalers,
                                    mva=self.mva)
        for i in range(self.test_pts):
            fd = {self.x: feats[i].reshape(1, self.history, -1),
                  self.y: targs[i].reshape(-1, 1),
                  self.keep_prob: 1}
            pred = self.sess.run(self.pred, feed_dict=fd)
            fut_df = pd.DataFrame({'close': pred.flatten()})
            resc_preds = dp.rescale_data(fut_df, self.scalers)
            ref_future_pred = dp.reform_prediction(self.rs[rs_start + i:rs_start + i + self.mva - 1],
                                                    resc_preds.close.values,
                                                    mva=self.mva)
            ref_future_pred = pd.DataFrame({'close': ref_future_pred})

            if last_preds is None:
                last_preds = ref_future_pred
            else:
                last_preds = last_preds.append(ref_future_pred)

        future_idx = self.sc_df.index[-last_preds.shape[0]:]
        last_preds = pd.DataFrame(last_preds)
        last_preds.set_index(future_idx, inplace=True)

        self.future_targs = future_targs
        self.future_preds = last_preds

        return future_targs, last_preds


    def pred_future_points(self):
        """
        Predicts the future points one at a time in the self.test_pts section.
        test_pts must be at least the size of the mva used to normalize,
        otherwise future_targs comes back all na's

        TODO: in the for loop, go through self.future points at a time,
        and reform the predictions in batches.
        """
        fut_preds = None
        start = -(self.test_pts + self.history + self.future)
        if self.mva is None:
            rs_start = -(self.test_pts - 1)
            ft_start = -(self.test_pts)
        else:
            rs_start = -(self.test_pts + self.mva - 1)
            ft_start = -(self.test_pts + self.mva)

        feats, targs = dp.create_feats_to_current(self.sc_df.iloc[start:],
                                                    history=self.history,
                                                    future=self.future)
        targ_df = pd.DataFrame({'close': targs.flatten()})
        future_targs = dp.reform_data(self.rs[ft_start:],
                                    targ_df,
                                    self.scalers,
                                    mva=self.mva)
        for i in range(self.test_pts):
            fd = {self.x: feats[i].reshape(1, self.history, -1),
                  self.y: targs[i].reshape(-1, 1),
                  self.keep_prob: 1}
            pred = self.sess.run(self.pred, feed_dict=fd)
            fut_df = pd.DataFrame({'close': pred.flatten()})
            if fut_preds is None:
                fut_preds = fut_df
            else:
                fut_preds = fut_preds.append(fut_df)

        resc_preds = dp.rescale_data(fut_preds, self.scalers)
        if self.mva is not None:
            # this method has the problem that small trends incur
            # positive feedback, sending the predictions wildly off-target.
            # don't recommend this for more than self.future points
            # ref_future_pred = dp.reform_future_predictions(self.rs[rs_start:-self.test_pts],
            #                                                resc_preds.close.values,
            #                                                mva=self.mva)

            # this method seems to just flatline the predictions...
            # ref_future_pred = dp.reform_future_predictions_mild(self.rs[rs_start:-self.test_pts],
            #                                                resc_preds.close.values,
            #                                                mva=self.mva)

            # the best way is to go through point by point, and use the reform_future_predictions
            # method to reform it up to self.future
            ref_future_pred = dp.reform_future_predictions(self.rs[rs_start:-self.test_pts],
                                                           resc_preds.close.values[:self.future],
                                                           mva=self.mva)

            for i in range(self.future, resc_preds.shape[0]):
                st = rs_start + i - self.future
                en = -self.test_pts + i - self.future
                temp_ref_future_pred = dp.reform_future_predictions(self.rs[st:en],
                                                                   resc_preds.close.values[i - self.future + 1:i + 1],
                                                                   mva=self.mva)
                ref_future_pred = ref_future_pred.append(temp_ref_future_pred.iloc[-1])
        else:
            ref_future_pred = resc_preds


        future_idx = self.sc_df.index[-ref_future_pred.shape[0]:]
        ref_future_pred.set_index(future_idx, inplace=True)
        future_targs.set_index(future_idx, inplace=True)

        self.future_targs = future_targs
        self.future_preds = ref_future_pred

        return future_targs, ref_future_pred


    def plot_future_preds(self, nb=False):
        if self.future_preds is None:
            _, _ = self.pred_future_points()

        if self.past_preds is None:
            _, _ = self.get_past_preds()

        start_idx = self.history + self.future - self.past - self.test_pts
        data = [go.Scatter(x=self.past_targs.index, y=self.past_targs.close, name='actual'),
               go.Scatter(x=self.past_preds.index, y=self.past_preds.close, name='predictions'),
               go.Scatter(x=self.future_targs.index, y=self.future_targs.close, name='actual future'),
               go.Scatter(x=self.future_preds.index, y=self.future_preds.close, name='future predictions')]
        layout = go.Layout(title=self.market)
        fig = go.Figure(data=data, layout=layout)
        iplot(fig) if nb else plot(fig)


    def score_preds(self, get_preds=False):
        """
        uses stepwise future predicts and targs to calculate MSE score
        """
        if get_preds:
            _, _ = self.pred_future_points()
            _, _ = self.get_past_preds()

        past_mse = tf.losses.mean_squared_error(self.past_preds, self.past_targs)
        past_loss = tf.reduce_mean(past_mse)
        past_l = self.sess.run(past_loss)
        self.past_scores.append(past_l)
        fut_mse = tf.losses.mean_squared_error(self.future_preds, self.future_targs)
        fut_loss = tf.reduce_mean(fut_mse)
        fut_l = self.sess.run(fut_loss)
        self.future_scores.append(fut_l)


    def step_thru_and_score(self):
        """
        steps through entire dataset in chunks of 'datapoints', and
        trains on datapoints - test_pts

        currently high loss on last chunk -- check it sometime
        probably the problem is in get_past_preds, because it's not indexing
        based on the current feats
        """
        if self.rs is None:
            print('loading/creating data...')
            self.create_data()

        num_chunks = self.rs.shape[0] // self.train_pts
        # start from the end and work backwards
        for i in range(1, num_chunks + 1):
            print('on chunk', str(i), 'of', str(num_chunks))
            start = self.sc_df.shape[0] - self.train_pts * i
            end = self.sc_df.shape[0] - self.train_pts * (i - 1)
            print(start, end)
            # need to double check this is working properly
            self.sc_df, self.scalers = dp.transform_data(self.rs[start:end],
                                                        mva=self.mva)
            self.feats, self.targs = dp.create_hist_feats(self.sc_df,
                                                        history=self.history,
                                                        future=self.future)
            self.train_feats = self.feats[:-self.test_pts]
            self.train_targs = self.targs[:-self.test_pts]

            # reset model and fit, then score
            self.create_graph()
            self.set_hyperparameters()
            self.train_net()
            self.score_preds(get_preds=True)



if __name__=="__main__":
    """
    experiment notes:
    when the number of training points is 27,000, the loss seems to be jumpy
    and actually mostly increase during trainin g, although the fits look ok.
    Decreasing the betas from 0.9 and 0.999 to 0.8 and 0.9 seems to improve
    stability, but the future predictions look much worse.

    When the number of training pts is 7,000, the loss consistently goes down,
    but the fits look about the same quality.

    The simple 3layer FC model seems to perform really well on 30k training points,
    although it trains kind of slow.  The future predictions are excellent.

    9-23-2017
    on 9k training points without mva normalization
    need a smaller learning rate, like 0.0001, especially for larger nets.
    3 and 8-layer nets won't even work withouth low lr.
    lr of 0.0001 and 100 epochs works for simple and 3-layer,

    the 8-layer just appears to be unstable no matter what...probably
    need to play with dropout and other optimizer hyperparameters
    """

    # this is for testing without a mva normalization
    models = ['simple_fc',
              'simple_3layer_fc',
              'simple_8layer_fc']
    # nn = nn_model(model=models[1])
    # #nn.step_thru_and_score()
    # nn.create_data()
    # nn.set_hyperparameters(epochs=200, lr=0.00005)
    # nn.create_graph()
    # nn.train_net()
    # nn.plot_future_preds()

    # this one works decently well

    # nn = nn_model(model=models[2], mva=30)
    # #nn.step_thru_and_score()
    # nn.create_data()
    # nn.set_hyperparameters(epochs=100, lr=0.0002)
    # nn.create_graph()
    # nn.train_net()
    # nn.plot_future_preds()


    # captures current spike up at 11 on sept 19 2017 in test data
    # hmm, missing the huge spike in predictions...
    # nn = nn_model(model=models[2], mva=30, train_pts=50000, test_pts=6500)
    # #nn.step_thru_and_score()
    # nn.create_data()
    # nn.set_hyperparameters(epochs=100, lr=0.0002)
    # nn.create_graph()
    # nn.train_net()
    # nn.plot_future_preds()

    # works ok but misses moves a lot
    # nn = nn_model(model=models[2], mva=30, train_pts=20000, test_pts=6500)
    # #nn.step_thru_and_score()
    # nn.create_data()
    # nn.set_hyperparameters(epochs=30, lr=0.0002)
    # nn.create_graph()
    # nn.train_net()
    # nn.plot_future_preds()

    # noticed a big move took 15 hours (900 mins) to complete
    # even the 8-layer dense model doesn't work well. Going to have to go LSTM
    # and add in number of buys/sells per trading block 
    nn = nn_model(model=models[2], mva=30, train_pts=40000, test_pts=6500, future=900)
    #nn.step_thru_and_score()
    nn.create_data()
    nn.set_hyperparameters(epochs=100, lr=0.0002)
    nn.create_graph()
    nn.train_net()
    nn.plot_future_preds()
