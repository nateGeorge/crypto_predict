from poloniex import Poloniex
import os
import time
import pandas as pd
from datetime import datetime
from threading import Thread

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)


def get_home_dir(repo_name='crypto_predict'):
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo_name]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir


def make_data_dirs():
    """
    Checks if data directory exists, if not, creates it.
    """
    exchanges = ['poloniex', 'bittrex']
    folds = ['order_books', 'trade_history']
    data_folders = ['data/' + f + '/' for f in folds]
    dds = ['data'] + data_folders + \
            [d + e for d in data_folders for e in exchanges]
    dirs = [HOME_DIR + d for d in dds]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)


HOME_DIR = get_home_dir()
make_data_dirs()

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
    datapath = HOME_DIR + 'data/order_books/poloniex/'
    buy_file = datapath + 'buy_orders_' + market + '.csv.gz'
    sell_file = datapath + 'sell_orders_' + market + '.csv.gz'
    if os.path.exists(buy_file):
        buy_df.to_csv(buy_file, compression='gzip', mode='a', header=False)
        sell_df.to_csv(sell_file, compression='gzip', mode='a', header=False)
    else:
        buy_df.to_csv(buy_file, compression='gzip')
        sell_df.to_csv(sell_file, compression='gzip')


def save_all_order_books():
    print('retrieving orderbooks...')
    buy_dfs, sell_dfs = get_all_orderbooks()
    print('done.')
    save_orderbooks(buy_dfs, sell_dfs)


def continuously_save_order_books(interval=60):
    """
    Saves all order books every 'interval' seconds.
    Poloniex allows 6 calls/second before your IP is banned.
    At one scrape every 60 seconds, this is going to get huge very fast.
    """
    def keep_saving():
        while True:
            save_all_order_books()
            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def get_trade_history(market='BTC_SC'):
    cur_ts = int(time.time())
    past = cur_ts - 60*60*24*7*4  # subtract 4 weeks
    h = polo.marketTradeHist(currencyPair=market, start=past, end=cur_ts)
    full_df = pd.io.json.json_normalize(h)
    full_df['date'] = pd.to_datetime(full_df['date'])
    earliest = 0
    cur_earliest = full_df.iloc[-1]['date'].value / 10**9
    while cur_earliest != earliest:
        earliest = cur_earliest
        past = earliest - 60*60*24*7*4  # subtract 4 weeks
        print('scraping another time...')
        start = time.time()
        h = polo.marketTradeHist(currencyPair=market, start=past, end=earliest)
        elapsed = time.time() - start
        # max api calls are 6/sec, don't want to get banned...
        if elapsed < 1/6.:
            print('scraping too fast, sleeping...')
            time.sleep(1/5. - elapsed)

        df = pd.io.json.json_normalize(h)
        df['date'] = pd.to_datetime(df['date'])
        full_df = full_df.append(df)
        cur_earliest = df.iloc[-1]['date'].value / 10**9

    # sometimes some duplicates
    full_df.drop_duplicates(inplace=True)
    full_df.set_index('date', inplace=True)
    # I like it sorted from oldest to newest
    full_df.sort_index()
    for col in ['amount', 'rate', 'total']:
        full_df[col] = pd.to_numeric(full_df[col])

    return full_df


def append_trade_history(market):
    """
    First checks what the latest date is in the dataframe, then scrapes the
    trade history from the current time back to then.
    """
    pass


def save_trade_history(df, market):
    """
    Saves a dataframe of the trade history for a market.
    """
    datapath = HOME_DIR + 'data/trade_history/poloniex/'
    filename = datapath + 'trade_history_' + market + '.csv.gz'
    df.to_csv(filename, compression='gzip')


def save_all_trade_history():
    ticks = polo.returnTicker()
    pairs = ticks.keys()
    for c in pairs:
        print('scaping', c)
        df = get_trade_history(c)
        print('saving', c)
        save_trade_history(df, c)




# TODO: get all trade history
# get all market depth and subscribe to updates, on major change (buy/sell)
# use marketTradeHist
# notify telegram bot etc
