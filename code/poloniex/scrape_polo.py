# core
import os
import sys
import time
from datetime import datetime
from threading import Thread
import traceback

# installed
# if running from the code/ folder, this will try to import
# a module Poloniex from the folder.  Better to run from within the
# poloniex folder as a result
from poloniex import Poloniex
import pandas as pd


def get_home_dir(repo='crypto_predict'):
    cwd = os.path.realpath(__file__)  # gets location of this file
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir

HOME_DIR = get_home_dir()
key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')
polo = Poloniex(key, sec)

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

make_data_dirs()
TRADE_DATA_DIR = HOME_DIR + 'data/trade_history/poloniex/'

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
    print('saving', market)
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


def recover_file(datafile, chunksize=1000000):
    """
    If EOFError comes up, reads as much of the dataframe as possible.
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
        df.to_csv(datafile, compression='gzip')
    else:
        print('file is already newest to oldest!')


def get_trade_history(market='BTC_BCN', two_h_delay=False, latest=None):
    """
    :param two_h_delay: if a 2 hour delay should be enacted between scrapings
    """
    # first check the latest date on data already there
    datafile = TRADE_DATA_DIR + market + '.csv.gz'
    latest_ts = None
    old_df = None
    if os.path.exists(datafile):
        # right now the csvs are saved as earliest data in the top
        # and latest data in the bottom.  Need to fix this, but for now
        # this code is not run
        earliest_to_latest = False  # instead, implemented another csv file that keeps track of latest scrape dates
        if earliest_to_latest:
            try:
                cur_df = pd.read_csv(datafile, index_col='date', parse_dates=['date'], chunksize=1)
                first = cur_df.get_chunk(1)
                latest = first.index[0].value / 10**9  # .value gets nanoseconds since epoch
            except EOFError:
                print('corrupted file, restoring from backup...')
                old_df = recover_file(datafile)
                latest = old_df.index[0].value / 10**9
        else:
            if latest is None:
                try:
                    old_df = pd.read_csv(datafile, index_col='date', parse_dates=['date'])
                except EOFError:
                    print('corrupted file, restoring from backup...')
                    old_df = recover_file(datafile)

                latest_ts = old_df.index[-1].value / 10**9
            else:
                latest_ts = latest.value / 10**9

        # get current timestamp in UTC...tradehist method takes utc times
        d = datetime.utcnow()
        epoch = datetime(1970, 1, 1)
        cur_ts = (d - epoch).total_seconds()
        if two_h_delay and (cur_ts - latest_ts) < 7200:
            print('scraped within last 2 hours, not scraping again...')
            return None, None, None
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
    h = polo.marketTradeHist(currencyPair=market, start=past, end=cur_ts)
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
    full_df.sort_index(inplace=True)
    for col in ['amount', 'rate', 'total']:
        full_df[col] = pd.to_numeric(full_df[col])

    return full_df, update, old_df


def save_trade_history(df, market, update, old_df=None):
    """
    Saves a dataframe of the trade history for a market.
    """
    filename = TRADE_DATA_DIR + market + '.csv.gz'
    if update:
        # TODO: need to get rid of reading the old DF, and just make_history_df
        # sure there are no overlapping points, then write csv with mode='a'
        if old_df is None:
            # used to do all this, but changed to appending because latest date is last
            # and can't scrape fast enough without appending
            # old_df = pd.read_csv(filename,
            #                     parse_dates=['date'],
            #                     infer_datetime_format=True)
            # old_df.set_index('date', inplace=True)
            # full_df = old_df.append(df)
            # full_df.drop_duplicates(inplace=True)
            # full_df.sort_index(inplace=True)
            df.to_csv(filename, compression='gzip', mode='a')
        else:
            full_df = old_df.append(df)
            full_df.drop_duplicates(inplace=True)
            full_df.sort_index(inplace=True)
            full_df.to_csv(filename, compression='gzip')
    else:
        df.to_csv(filename, compression='gzip')


def save_all_trade_history(two_h_delay=False):
    lat_scr_file = '/'.join(TRADE_DATA_DIR.split('/')[:-2] + ['']) + 'latest_polo_scrape_dates.csv'
    lat_scr_df = None
    # if os.path.exists(lat_scr_file):
    #     try:
    #         lat_scr_df = pd.read_csv(lat_scr_file, index_col='market', parse_dates=['latest'])
    #     except EOFError:
    #         lat_scr_df = None

    ticks = polo.returnTicker()
    pairs = sorted(ticks.keys())
    for c in pairs:
        print('checking', c)
        if lat_scr_df is None:
            df, update, old_df = get_trade_history(c, two_h_delay=two_h_delay)
        else:
            if c in lat_scr_df.index:
                df, update, old_df = get_trade_history(c, two_h_delay=two_h_delay, latest=lat_scr_df.loc[c]['latest'])
            else:
                df, update, old_df = get_trade_history(c, two_h_delay=two_h_delay)

        if df is not None:
            print('saving', c)
            save_trade_history(df, c, update, old_df)
            if lat_scr_df is None:
                lat_scr_df = pd.DataFrame({'market':[c], 'latest':[df.index[-1]]})
                lat_scr_df.set_index('market', inplace=True)
            elif c in lat_scr_df.index:
                lat_scr_df[c] = df.index[-1]
            else:
                lat_df = pd.DataFrame({'market':[c], 'latest':[df.index[-1]]})
                lat_df.set_index('market', inplace=True)
                lat_scr_df = lat_scr_df.append(lat_df)
                lat_scr_df.to_csv(lat_scr_file)


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
    # updates all trade histories
    save_all_trade_history()

    # going to have to go through and remove dupes...need to pass latest row instead of latest date only, so can stop the new stuff there
