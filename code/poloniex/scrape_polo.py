# core
import os
import gc
import sys
import time
import csv

from datetime import datetime, timedelta
from threading import Thread
import traceback
from io import StringIO
from poloniex import PoloniexError

# installed
# if running from the code/ folder, this will try to import
# a module Poloniex from the folder.  Better to run from within the
# poloniex folder as a result
# should be the poloniexapi package, not poloniex (when installing with pip)
# from here: https://github.com/s4w3d0ff/python-poloniex
from poloniex import Poloniex
import poloniex
import pandas as pd
import feather as ft
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import \
    ARRAY, BIGINT, BIT, BOOLEAN, BYTEA, CHAR, CIDR, DATE, \
    DOUBLE_PRECISION, ENUM, FLOAT, HSTORE, INET, INTEGER, \
    INTERVAL, JSON, JSONB, MACADDR, MONEY, NUMERIC, OID, REAL, SMALLINT, TEXT, \
    TIME, TIMESTAMP, UUID, VARCHAR, INT4RANGE, INT8RANGE, NUMRANGE, \
    DATERANGE, TSRANGE, TSTZRANGE, TSVECTOR


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

trade_history_dtypes = {'amount': DOUBLE_PRECISION,
                        'date': TIMESTAMP,
                        'globalTradeID': BIGINT,
                        'rate': DOUBLE_PRECISION,
                        'total': DOUBLE_PRECISION,
                        'type' : BOOLEAN}


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
        if not os.path.exists(d):
            print('making', d)
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
        except PoloniexError:
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

    del orderbooks
    del sell_headers
    del buy_headers
    gc.collect()

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

    save_orderbooks(buy_dfs, sell_dfs)
    print('done.')

    del buy_dfs
    del sell_dfs
    gc.collect()


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


def load_order_book(market='BTC_CLAM'):
    datapath = HOME_DIR + 'data/order_books/poloniex/'
    buy_file = datapath + 'buy_orders_' + market + '.csv.gz'
    sell_file = datapath + 'sell_orders_' + market + '.csv.gz'
    buy_df = pd.read_csv(buy_file)
    sell_df = pd.read_csv(sell_file)
    return buy_df, sell_df


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
        except PoloniexError:
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
        except PoloniexError:
            tries += 1
            time.sleep(1)

        if tries == 100:
            print("couldn't get trade history")
            return None
            break


def try_to_get_dates():
    # get earliest trade to compare with existing data
    import pytz
    market = 'USDT_BTC'
    query = 'SELECT * FROM {} ORDER BY globaltradeid LIMIT 1;'.format(market.lower())
    query = 'SELECT * FROM {} WHERE globaltradeid = 7136498;'.format(market.lower())
    earliest_point = engine.execute(query).fetchone()
    earliest_ts = datetime.timestamp(earliest_point[1].astimezone(pytz.utc))
    h = get_polo_hist(market, start=earliest_ts, end=earliest_ts + 20000)


def get_trade_history_old(market='BTC_AMP', two_h_delay=False, latest=None):
    """
    Saves trade history to csv.gz file.

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
    # new: orderNumber is an unknown thing, not in docs
    # tradeID as well as global trade id...drop the ordernumber for now
    full_df.drop('orderNumber', axis=1, inplace=True)
    full_df.drop('tradeID', axis=1, inplace=True)

    if full_df.shape[0] == 0:
        print('no data, skipping')
        del full_df
        del h
        gc.collect()
        return None, None
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
    dfs = []
    while cur_earliest != earliest and cur_earliest > very_earliest:
        earliest = cur_earliest
        past = earliest - 60*60*24*7*4  # subtract 4 weeks
        print('scraping another time...')
        start = time.time()
        h = get_polo_hist(market=market, start=past, end=earliest)
        if h is None:
            print("getting history choked")
            return None, None

        df = pd.io.json.json_normalize(h)
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)
        # full_df = full_df.append(df)
        cur_earliest = df.iloc[-1]['date'].value / 10**9

        elapsed = time.time() - start
        # max api calls are 6/sec, don't want to get banned...
        if elapsed < 1/6.:
            print('scraped in', elapsed)
            print('scraping too fast, sleeping for', 1/5. - elapsed)
            time.sleep(1/5. - elapsed)

        print('took', time.time() - start, 's to fully scrape one point')



    full_df = pd.concat(dfs)
    # find where we should cutoff new data
    full_df.sort_values(by='tradeID', inplace=True)
    full_df.reset_index(inplace=True, drop=True)
    full_df['date'] = pd.to_datetime(full_df['date'], utc=True)
    if latest is not None:
        latest_idx = full_df[full_df['globalTradeID'] == old_df['globalTradeID']].index[0]
        # take everything from the next trade on
        full_df = full_df.iloc[latest_idx + 1:]

    del old_df
    del h
    gc.collect()

    if full_df.shape[0] > 0:
        # sometimes some duplicates  -- don't think we need this though, oh well
        full_df.drop_duplicates(inplace=True)
        # sorted from oldest at the top to newest at bottom for now
        for col in ['amount', 'rate', 'total']:
            full_df[col] = pd.to_numeric(full_df[col])

        return full_df, update
    else:
        return None, None


def get_trade_history(market='USDC_GRIN', two_h_delay=False):
    """
    USDC_GRIN is best to test with for now because it dosen't have lots of data.

    Saves trade history to PSQL database.

    :param two_h_delay: if a 2 hour delay should be enacted between scrapings
    """
    # get connection to DB
    engine = create_sql_connection()
    #conn = engine.connect()
    tables = engine.table_names()

    # get latest date if any data exists
    latest_ts = None
    no_data = True
    if market.lower() in tables:
        # right now the csvs are saved as earliest data in the top
        query = 'SELECT * FROM {} ORDER BY date DESC LIMIT 1;'.format(market.lower())
        #columns = conn.execute(query).keys()
        latest_point = engine.execute(query).fetchone()  # get latest datapoint
        if latest_point is None:
            print("no data in table")
        else:
            no_data = False

            latest_ts = datetime.timestamp(latest_point[1])

            # get current timestamp in UTC...tradehist method takes utc times
            d = datetime.utcnow()
            epoch = datetime(1970, 1, 1)
            cur_ts = (d - epoch).total_seconds()
            if two_h_delay and (cur_ts - latest_ts) < 7200:
                print('scraped within last 2 hours, not scraping again...')
                engine.dispose()
                return None, None
            else:
                print('scraping updates')
                update = True

    if no_data:
        print('scraping new, no table exists')
        create_table(engine, market)
        update = False
        # get current timestamp in UTC...tradehist method takes utc times
        d = datetime.utcnow()
        epoch = datetime(1970, 1, 1)
        cur_ts = (d - epoch).total_seconds()

    # don't need psql connection after this
    engine.dispose()

    # get past time, subtract 4 weeks
    past = cur_ts - 60*60*24*7*4
    h = get_polo_hist(market=market, start=past, end=cur_ts)
    if h is None:
        print("getting history choked")
        return None, None

    full_df = pd.io.json.json_normalize(h)
    if full_df.shape[0] == 0:
        print('no data, skipping')
        del full_df
        del h
        gc.collect()
        return None, None

    # not sure why this is here...
    full_df = clean_df(full_df)

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
    dfs = []
    while cur_earliest != earliest and cur_earliest > very_earliest:
        earliest = cur_earliest
        past = earliest - 60*60*24*7*4  # subtract 4 weeks
        print('scraping another time...')
        start = time.time()
        h = get_polo_hist(market=market, start=past, end=earliest)
        if h is None:
            print("getting history choked")
            return None, None

        df = pd.io.json.json_normalize(h)
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)
        # full_df = full_df.append(df)
        cur_earliest = df.iloc[-1]['date'].value / 10**9

        elapsed = time.time() - start
        # max api calls are 6/sec, don't want to get banned...
        if elapsed < 1/6.:
            print('scraped in', elapsed)
            print('scraping too fast, sleeping for', 1/5. - elapsed)
            time.sleep(1/5. - elapsed)

        print('took', time.time() - start, 's to fully scrape one point')




    del h
    gc.collect()

    if len(dfs) > 0: #full_df.shape[0] > 0:
        full_df = pd.concat(dfs)
        full_df = clean_df(full_df)
        # find where we should cutoff new data
        full_df.sort_values(by='globaltradeid', inplace=True)
        full_df.reset_index(inplace=True, drop=True)
        # sometimes some duplicates  -- don't think we need this though, oh well
        full_df.drop_duplicates(inplace=True)
        # sorted from oldest at the top to newest at bottom for now
        for col in ['amount', 'rate', 'total']:
            full_df[col] = pd.to_numeric(full_df[col])

        return full_df, update
    else:
        return None, None


def clean_df(df):
    """
    cleans up raw data from poloniex
    """
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_localize('UTC')
    df.drop('tradeID', axis=1, inplace=True)
    df.drop('orderNumber', axis=1, inplace=True)
    df['type'] = df['type'].apply(convert_buy_sell_boolean)
    df['type'] = df['type'].astype('bool')
    df.columns = [c.lower() for c in df.columns]

    return df


def save_trade_history_hdf(df, market, update):
    """
    Saves a dataframe of the trade history for a market.
    """
    filename = TRADE_DATA_DIR + market + '.hdf5'
    if update:
        df.to_hdf(filename, 'data', mode='a', complib='blosc', complevel=9, format='table', append=True)
    else:
        df.to_hdf(filename, 'data', mode='w', complib='blosc', complevel=9, format='table')


def save_trade_history_psql(df, market):
    """
    Saves a dataframe of the trade history for a market.
    """
    engine = create_sql_connection()
    df.to_sql(market.lower(), engine, if_exists='append', index=False, method=psql_insert_copy, chunksize=10000000, dtype=trade_history_dtypes)
    engine.dispose()


def save_all_trade_history_hdf(two_h_delay=False):
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
            del df
            del update
            gc.collect()

    end = time.time()

    print('done!  took', int(end-start), 'seconds')
    del ticks
    del pairs
    gc.collect()


def save_all_trade_history_psql(two_h_delay=False):
    start = time.time()
    ticks = get_tickers()
    if ticks is None:
        print("couldn't get tickers")
        return

    pairs = sorted(ticks.keys())
    for c in pairs:
        if 'EOS' in c:
            print("EOS was disabled for trading and threw error; skipping")
            continue

        print('checking', c)
        df, update = get_trade_history(c, two_h_delay=two_h_delay)

        if df is not None:
            print('saving', c)
            save_trade_history_psql(df, c)
            del df
            del update
            gc.collect()

    end = time.time()

    print('done!  took', int(end-start), 'seconds')
    backup_db()
    del ticks
    del pairs
    gc.collect()


def continuously_save_trade_history_old(interval=600):
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


def continuously_save_trade_history(interval=600):
    """
    Saves all order books every 'interval' seconds to PSQL.
    Poloniex allows 6 calls/second before your IP is banned/blocked.
    """
    def keep_saving():
        while True:
            try:
                save_all_trade_history_psql()
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


def load_trade_history(market='USDT_BTC', format='ft'):
    """
    Loads trade history from hdf5 or feather file.

    market: string, currency pair
    format: string, ft for feather or hdf for hdf5
    """
    if format == 'hdf':
        datafile = TRADE_DATA_DIR + market + '.hdf5'
        df = pd.read_hdf(datafile)
    elif format == 'ft':
        datafile = TRADE_DATA_DIR + market + '.ft'
        df = pd.read_feather(datafile)

    return df


def convert_to_sql(market='USDT_BTC', format='ft'):
    """
    Moves data to SQL database from feather or hdf5 file.  Not finished, should probably delete.
    """
    if format == 'hdf':
        datafile = TRADE_DATA_DIR + market + '.hdf5'
        df = pd.read_hdf(datafile)
    elif format == 'ft':
        datafile = TRADE_DATA_DIR + market + '.ft'
        df = pd.read_feather(datafile)


def create_sql_connection(remote=False, db='poloniex_trade_history'):
    """
    Creates connection to SQL database
    """
    user = os.environ.get('psql_username')
    passwd = os.environ.get('psql_pass')
    engine = create_engine('postgresql://{}:{}@cerium:5432/{}'.format(user, passwd, db))
    return engine


def create_first_table(engine, market='USDT_BTC', db='poloniex_trade_history'):
    """
    Creates first table in db for market pair.  After the first table, others can inherit the columns and datatypes.
    """
    with engine.connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS {} (
                       amount double precision,
                       date timestamptz,
                       globalTradeID bigint,
                       rate double precision,
                       total double precision,
                       type boolean
                       )""".format(market))


def create_all_tables(engine):
    """
    Creates all tables for possible datasets on poloniex.
    """
    ticks = get_tickers()
    pairs = sorted(ticks.keys())
    for p in pairs:
        create_table(engine, p)


def create_table(engine, market):
    with engine.connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS {} (
                       amount double precision,
                       date timestamptz,
                       globalTradeID bigint,
                       rate double precision,
                       total double precision,
                       type boolean
                       )""".format(market))

        con.execute("""SET TIMEZONE TO 'UTC';""")


def convert_buy_sell_boolean(x):
    if x == 'buy':
        return 1
    elif x == 'sell':
        return 0


def move_feather_to_sql(engine):
    """
    Moves all feather files to the SQL database.  Currently just gets active pairs.
    """
    ticks = get_tickers()
    pairs = sorted(ticks.keys())
    for p in pairs:
        print(p)
        datafile = TRADE_DATA_DIR + market + '.ft'
        df = pd.read_feather(datafile)
        # get rid of unnecessary column; convert type (buy/sell) to boolean
        df.drop('tradeID', axis=1, inplace=True)
        df['type'] = df['type'].apply(convert_buy_sell_boolean)
        df['type'] = df['type'].astype('bool')
        df.columns = [c.lower() for c in df.columns]

        start = time.time()
        df.to_sql(p.lower(), engine, if_exists='append', index=False, method=psql_insert_copy, chunksize=10000000, dtype=trade_history_dtypes)
        end = time.time()
        print('took', end - start, 'seconds')


def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def backup_db():
    """
    exports backup of database
    """
    tz = pytz.timezone('UTC')
    todays_date_utc = datetime.now(tz).strftime('%m-%d-%Y')
    filename = '/home/nate/Dropbox/data/postgresql/crypto/poloniex_trade_history.{}.pgsql'.format(todays_date_utc)

    pg_pass = os.environ.get('postgres_pass')
    os.system('export PGPASSWORD=' + pg_pass)
    os.system('pg_dump -U nate rss_feeds > ' + filename)
    # remove old files
    list_of_files = glob.glob('/home/nate/Dropbox/data/postgresql/crypto/poloniex_trade_history.*.pgsql')
    latest_file = max(list_of_files, key=os.path.getctime)
    for f in list_of_files:
        if f != latest_file:
            os.remove(f)


# TODO: get all trade history
# get all market depth and subscribe to updates, on major change (buy/sell)
# use marketTradeHist
# notify telegram bot etc

if __name__ == "__main__":
    pass
    # updates all trade histories
    #save_all_trade_history()
