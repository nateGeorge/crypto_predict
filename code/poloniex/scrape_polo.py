# core
import os
import sys
import time
from datetime import datetime, timedelta
from threading import Thread
import traceback

# installed
# if running from the code/ folder, this will try to import
# a module Poloniex from the folder.  Better to run from within the
# poloniex folder as a result
from poloniex import Poloniex
import poloniex
import pandas as pd
import feather as ft


def get_home_dir(repo='crypto_predict'):
    cwd = os.path.realpath(__file__)  # gets location of this file
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir

HOME_DIR = '/media/nate/data_lake/crytpo_predict/'#get_home_dir()
key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)

def make_data_dirs():
    """
    Checks if data directory exists, if not, creates it.
    """
    exchanges = ['poloniex', 'bittrex']
    folds = ['order_books', 'trade_history', '550h_trade_history']
    data_folders = ['data/' + f + '/' for f in folds]
    dds = ['data'] + data_folders + \
            [d + e for d in data_folders for e in exchanges]
    dirs = [HOME_DIR + d for d in dds]
    for d in dirs:
        print(d)
        if not os.path.exists(d):
            os.mkdir(d)

make_data_dirs()
TRADE_DATA_DIR = HOME_DIR + 'data/trade_history/poloniex/'
SM_TRADE_DATA_DIR = HOME_DIR + 'data/550h_trade_history/poloniex/'
CSV_WRITE_CHUNK = 500000  # chunksize for writing csv...doesn't seem to make a difference

def get_all_orderbooks():
    """
    returns dicts of pandas dataframes with all currency pair orderbooks,
    full depth
    """
    # returns dict with currencyPair as primary keys, then 'asks', 'bids'
    # 'isFrozen', 'seq' - seq is the sequence number for the push api
    tries = 1
    while True:
        try:
            orderbooks = polo.returnOrderBook(currencyPair='all', depth=1000000)
            break
        except polo.PoloniexError:
            tries += 1
            time.sleep(1)

        if tries == 100:
            print('unable to get orderbooks')
            return None, None
            break  # probably not necessary, but just in case

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
    print('saving', market)
    if os.path.exists(buy_file):
        buy_df.to_csv(buy_file, compression='gzip', mode='a', header=False, chunksize=CSV_WRITE_CHUNK)
        sell_df.to_csv(sell_file, compression='gzip', mode='a', header=False, chunksize=CSV_WRITE_CHUNK)
    else:
        buy_df.to_csv(buy_file, compression='gzip', chunksize=CSV_WRITE_CHUNK)
        sell_df.to_csv(sell_file, compression='gzip', chunksize=CSV_WRITE_CHUNK)


def save_all_order_books():
    print('retrieving orderbooks...')
    buy_dfs, sell_dfs = get_all_orderbooks()
    if buy_dfs is None:
        print('unable to get orderbooks, trying one more time')
        buy_dfs, sell_dfs = get_all_orderbooks()
        if buy_dfs is None:
            print('still couldn\'t get orderbooks...')
            return

    print('done.')
    save_orderbooks(buy_dfs, sell_dfs)


def continuously_save_order_books(interval=600):
    """
    Saves all order books every 'interval' seconds.
    Poloniex allows 6 calls/second before your IP is banned.
    """
    def keep_saving():
        while True:
            try:
                save_all_order_books()
            except:
                traceback.print_exc()

            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def recover_file(datafile, chunksize=1000000):
    """
    If EOFError comes up, reads as much of the dataframe as possible.
    This will happen if the computer is shut down while writing data.
    """
    full_df = None
    skip = 0
    while True:
        try:
            if skip == 0:
                cur_df = pd.read_csv(datafile, chunksize=chunksize, index_col='date', parse_dates=['date'])
            else:
                cur_df = pd.read_csv(datafile, chunksize=chunksize, index_col='date', parse_dates=['date'], skiprows=range(1, skip))
            for c in cur_df:
                if full_df is None:
                    full_df = c
                else:
                    full_df = full_df.append(c)
        except EOFError:  # eventually we will hit this when we get to the corrupted part
            if full_df is not None:
                skip = full_df.shape[0]

            if chunksize == 1:
                return full_df

            chunksize = chunksize // 2
            if chunksize == 0:
                chunksize = 1


def convert_earliest_to_latest(market):
    """
    takes a file that is latest to earliest and flips it

    might not want to do this actually, because then restoring earliest data from corrupted files will be trickier
    """
    datafile = TRADE_DATA_DIR + market + '.csv.gz'
    try:
        old_df = pd.read_csv(datafile, index_col='date', parse_dates=['date'])
    except EOFError:
        print('corrupted file, restoring from backup...')
        old_df = recover_file(datafile)

    first_date = old_df.index[0]
    last_date = old_df.index[-1]

    if last_date > first_date:  # it is from oldest to newest
        df = old_df.iloc[::-1].copy()
        df.to_csv(datafile, compression='gzip', chunksize=CSV_WRITE_CHUNK)
    else:
        print('file is already newest to oldest!')


def remove_dupes(market='BTC_AMP'):
    """
    pretty self-explanatory
    """
    datafile = TRADE_DATA_DIR + market + '.hdf5'
    old_df = pd.read_hdf(datafile)
    dd_df = old_df.drop_duplicates()
    num_dupes = old_df.shape[0] - dd_df.shape[0]
    if num_dupes == 0:
        print('no dupes, skipping...')
        return

    dd_df.sort_index(inplace=True)
    dd_df.to_hdf(datafile, 'data', mode='w', complib='blosc', complevel=9, format='table')


def get_tickers():
    tries = 1
    while True:
        try:
            ticks = polo.returnTicker()
            return ticks
            break
        except polo.PoloniexError:
            tries += 1
            time.sleep(1)

        if tries == 100:
            print("couldn't get tickers")
            return None
            break

def remove_all_dupes():
    ticks = get_tickers()
    if ticks is None:
        print("couldn't get tickers")
        return

    pairs = sorted(ticks.keys())
    for c in pairs:
        print('cleaning', c)
        remove_dupes(c)


def check_for_dupes(market='BTC_AMP'):
    datafile = TRADE_DATA_DIR + market + '.hdf5'
    old_df = pd.read_hdf(datafile)
    dd_df = old_df.drop_duplicates()
    print(old_df.shape[0] - dd_df.shape[0], 'dupes')


def convert_ft_hdf5(market='BTC_AMP'):
    ft_datafile = TRADE_DATA_DIR + market + '.ft'
    hdf_datafile = TRADE_DATA_DIR + market + '.hdf5'
    old_df = ft.read_dataframe(ft_datafile)
    old_df.to_hdf(hdf_datafile, 'data', mode='w', complib='blosc', complevel=9, format='table')


def convert_all_to_hdf5():
    ticks = get_tickers()
    if ticks is None:
        print("couldn't get tickers")
        return

    pairs = sorted(ticks.keys())
    for c in pairs:
        print('converting to hdf5:', c)
        convert_ft_hdf5(market=c)


def get_polo_hist(market, start, end):
    tries = 1
    while True:
        try:
            h = polo.marketTradeHist(currencyPair=market, start=start, end=end)
            return h
            break
        except polo.PoloniexError:
            tries += 1
            time.sleep(1)

        if tries == 100:
            print("couldn't get trade history")
            return None
            break

def get_trade_history(market='BTC_AMP', two_h_delay=False, latest=None):
    """
    :param two_h_delay: if a 2 hour delay should be enacted between scrapings
    :param latest: pandas series with latest trade datapoint in csv
    """
    # first check the latest date on data already there
    datafile = TRADE_DATA_DIR + market + '.hdf5'
    print(datafile)
    latest_ts = None
    old_df = None
    if os.path.exists(datafile):
        # right now the csvs are saved as earliest data in the top
        old_df = pd.read_hdf(datafile, start=-1)  # read last row only
        latest_ts = old_df.iloc[-1]['date'].value / 10**9

        # get current timestamp in UTC...tradehist method takes utc times
        d = datetime.utcnow()
        epoch = datetime(1970, 1, 1)
        cur_ts = (d - epoch).total_seconds()
        if two_h_delay and (cur_ts - latest_ts) < 7200:
            print('scraped within last 2 hours, not scraping again...')
            return None, None
        else:
            print('scraping updates')
            update = True
    else:
        print('scraping new, no file exists')
        update = False
        # get current timestamp in UTC...tradehist method takes utc times
        d = datetime.utcnow()
        epoch = datetime(1970, 1, 1)
        cur_ts = (d - epoch).total_seconds()

    # get past time, subtract 4 weeks
    past = cur_ts - 60*60*24*7*4
    h = get_polo_hist(market=market, start=past, end=cur_ts)
    if h is None:
        print("getting history choked")
        return None, None

    full_df = pd.io.json.json_normalize(h)
    full_df['date'] = pd.to_datetime(full_df['date'])
    # very_earliest keeps track of the last date in the saved df on disk
    if latest_ts is None:
        very_earliest = 0
    else:
        very_earliest = latest_ts

    earliest = 0
    cur_earliest = full_df.iloc[-1]['date'].value / 10**9
    # if we get to the start of the data, quit, or if the earliest currently
    # scraped date is less than the earliest in the saved df on disk, break
    # the loop
    while cur_earliest != earliest and cur_earliest > very_earliest:
        earliest = cur_earliest
        past = earliest - 60*60*24*7*4  # subtract 4 weeks
        print('scraping another time...')
        start = time.time()
        h = get_polo_hist(market=market, start=past, end=earliest)
        if h is None:
            print("getting history choked")
            return None, None

        elapsed = time.time() - start
        # max api calls are 6/sec, don't want to get banned...
        if elapsed < 1/6.:
            print('scraping too fast, sleeping...')
            time.sleep(1/5. - elapsed)

        df = pd.io.json.json_normalize(h)
        df['date'] = pd.to_datetime(df['date'])
        full_df = full_df.append(df)
        cur_earliest = df.iloc[-1]['date'].value / 10**9

    # find where we should cutoff new data
    full_df.sort_values(by='tradeID', inplace=True)
    full_df.reset_index(inplace=True, drop=True)
    if latest is not None:
        latest_idx = full_df[full_df['globalTradeID'] == old_df['globalTradeID']].index[0]
        # take everything from the next trade on
        full_df = full_df.iloc[latest_idx + 1:]
    if full_df.shape[0] > 0:
        # sometimes some duplicates  -- don't think we need this though, oh well
        full_df.drop_duplicates(inplace=True)
        # sorted from oldest at the top to newest at bottom for now
        for col in ['amount', 'rate', 'total']:
            full_df[col] = pd.to_numeric(full_df[col])

        return full_df, update
    else:
        return None, None


def save_trade_history(df, market, update):
    """
    Saves a dataframe of the trade history for a market.
    """
    filename = TRADE_DATA_DIR + market + '.hdf5'
    if update:
        df.to_hdf(filename, 'data', mode='a', complib='blosc', complevel=9, format='table', append=True)
    else:
        df.to_hdf(filename, 'data', mode='w', complib='blosc', complevel=9, format='table')


def save_all_trade_history(two_h_delay=False):
    start = time.time()
    ticks = get_tickers()
    if ticks is None:
        print("couldn't get tickers")
        return

    pairs = sorted(ticks.keys())
    for c in pairs:
        if 'EOS' in c:
            print("EOS currently disabled for trading and throws error; skipping")
            continue

        print('checking', c)
        df, update = get_trade_history(c, two_h_delay=two_h_delay)

        if df is not None:
            print('saving', c)
            save_trade_history(df, c, update)

    end = time.time()

    print('done!  took', int(end-start), 'seconds')


def continuously_save_trade_history(interval=600):
    """
    Saves all order books every 'interval' seconds.
    Poloniex allows 6 calls/second before your IP is banned.
    """
    def keep_saving():
        while True:
            try:
                save_all_trade_history()
            except:
                traceback.print_exc()

            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def get_all_loans():
    """
    """
    pass


def get_loans(m='BTC_ETH'):
    """
    """
    pass



# TODO: get all trade history
# get all market depth and subscribe to updates, on major change (buy/sell)
# use marketTradeHist
# notify telegram bot etc

if __name__ == "__main__":
    pass
    # updates all trade histories
    #save_all_trade_history()
