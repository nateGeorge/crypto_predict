from poloniex import Poloniex
import os
import pandas as pd
from datetime import datetime
from threading import Thread

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)


def get_home_dir():
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == 'crytpo_predict']
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir


def get_all_orderbooks():
    """
    returns dicts of pandas dataframes with all currency pair orderbooks,
    full depth
    """
    # returns dict with currencyPair as primary keys, then 'asks', 'bids'
    # 'isFrozen', 'seq' - seq is the sequence number for the push api
    orderbooks = polo.returnOrderBook(currencyPair='all', depth=1000000)
    timestamp = pd.to_datetime(datetime.now())
    sell_dfs = {}
    buy_dfs = {}
    sell_headers = ['price', 'amt']
    buy_headers = ['price', 'amt']
    for c in orderbooks:
        sell_dfs[c] = pd.DataFrame(orderbooks[c]['asks'],
                                    columns=sell_headers)
        buy_dfs[c] = pd.DataFrame(orderbooks[c]['bids'],
                                    columns=buy_headers)
        sell_dfs[c]['timestamp'] = timestamp
        buy_dfs[c]['timestamp'] = timestamp
        sell_dfs[c].set_index('timestamp', inplace=True)
        buy_dfs[c].set_index('timestamp', inplace=True)

    return buy_dfs, sell_dfs


def save_orderbooks(buy_dfs, sell_dfs):
    """
    Saves all orderbooks at a given time
    """
    for c in buy_dfs.keys():
        save_orderbook(buy_dfs[c], sell_dfs[c], c)


def save_orderbook(buy_df, sell_df, market):
    """
    Saves one orderbook, in separate buy and sell files
    """
    buy_file = HOME_DIR + 'data/order_books/poloniex/buy_orders_' + market + '.csv.gz'
    sell_file = HOME_DIR + 'data/order_books/poloniex/sell_orders_' + market + '.csv.gz'
    if os.path.exists(buy_file):
        buy_df.to_csv(buy_file, compression='gzip', mode='a', header=False)
        sell_df.to_csv(sell_file, compression='gzip', mode='a', header=False)
    else:
        buy_df.to_csv(buy_file, compression='gzip')
        sell_df.to_csv(sell_file, compression='gzip')


def save_all_order_books():
    buy_dfs, sell_dfs = get_all_orderbooks()
    save_orderbooks(buy_dfs, sell_dfs)


def continuously_save_order_books(interval=600):
    """
    Saves all order books every 'interval' seconds.
    Poloniex allows 6 calls/second before your IP is banned.
    """
    def keep_saving():
        while True:
            save_all_order_books()
            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def get_trade_history():
    """
    """

# TODO: get all trade history
# get all market depth and subscribe to updates, on major change (buy/sell)
# use marketTradeHist
# notify telegram bot etc
