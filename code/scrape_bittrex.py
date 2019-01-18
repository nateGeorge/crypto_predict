# core
import os
import gc
import re
import time
from datetime import datetime
from threading import Thread

# installed
import pandas as pd
import requests
import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
# for writing to sql with pandas
from sqlalchemy import create_engine


PG_UNAME = os.environ.get('postgres_uname')
PG_PASS = os.environ.get('postgres_pass')
TH_DB = 'bittrex'
# create db if not already there
# check if db exists

def create_db_conn():
    try:
        conn = pg.connect(dbname=TH_DB, user=PG_UNAME, password=PG_PASS)
    except pg.OperationalError:
        conn = pg.connect(dbname='postgres', user=PG_UNAME, password=PG_PASS)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute('CREATE DATABASE ' + TH_DB)
        cur.close()
        conn.close()
        conn = pg.connect(dbname=TH_DB, user='nate', password=PG_PASS)

    return conn


def get_all_tables():
    # gets list of all tables
    cursor = conn.cursor()
    cursor.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
    return cursor.fetchall()


def get_home_dir():
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == 'crypto_predict']
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir


def get_all_currency_pairs(show_mkts=False):
    while True:  # in case of ssl error
        try:
            res = requests.get('https://bittrex.com/api/v1.1/public/getmarkets')
            break
        except Exception as e:
            print(e)
            time.sleep(10)

    if res.json()['success']:
        markets = res.json()['result']
        market_names = []
        for m in markets:
            if show_mkts:
                print(m['MarketName'])
            market_names.append(m['MarketName'])

        return sorted(market_names)
    else:
        print('error! ', res.json()['message'])
        return None


HOME_DIR = '/media/nate/data_lake/crytpo_predict/'#get_home_dir()
MARKETS = get_all_currency_pairs()


def get_all_summaries():
    while True:  # just in case of SSL error
        try:
            res = requests.get('https://bittrex.com/api/v1.1/public/getmarketsummaries')
            break
        except Exception as e:
            print(e)
            time.sleep(10)

    if res.json()['success']:
        summary = res.json()['result']
        return summary
    else:
        print('error! ', res.json()['message'])
        return None


def get_all_tickers():
    tickers = []
    for m in MARKETS:
        while True:
            try:
                res = requests.get('https://bittrex.com/api/v1.1/public/getticker?market=' + m)
                break
            except Exception as e:
                print(e)
                time.sleep(10)

        if res.json()['success']:
            t = res.json()['result']
            if t is None:
                print('error for', m + '!', 'result was None. Message:', res.json()['message'])
                continue

            t['MarketName'] = m
            tickers.append(t)
        else:
            print('error for', m + '!', res.json()['message'])

    df = pd.io.json.json_normalize(tickers)
    df.set_index('MarketName', inplace=True)
    return df


def get_trade_history(market):
    tries = 0
    while True:  # sometimes an SSL connection error...just wait a few seconds and try again
        tries += 1
        if tries == 6:
            return None
        try:
            res = requests.get('https://bittrex.com/api/v1.1/public/getmarkethistory?market=' + market)
            break
        except Exception as e:
            print(e)
            time.sleep(10)

    try:
        if res.json()['success']:
            history = res.json()['result']
            return history
        else:
            print('error! ', res.json()['message'])
            return None
    except Exception as e:
        print('exception! error!')
        print(e)
        return None


def save_all_trade_history():
    for m in MARKETS:
        print('saving', m, 'trade history')
        history = get_trade_history(m)
        if history is None or len(history) == 0:
            print('no history!')
            continue

        df = make_history_df(history)
        filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', m) + '.csv.gz'
        if os.path.exists(filename):
            old_df = pd.read_csv(filename, index_col='TimeStamp')
            full_df = old_df.append(df)
            full_df.drop_duplicates(inplace=True)
        else:
            full_df = df

        full_df.to_csv(filename, compression='gzip')
        # pause 5s to allow for graceful shutdown for now
        del history
        del df
        try:
            del old_df
        except NameError:
            pass
        del full_df
        gc.collect()
        print('done saving; resting 5s')
        time.sleep(2)

    print('\n\ndone!\n\n')


def save_all_trade_history_old():
    """
    saves data to CSVs...pretty inefficient
    """
    for m in MARKETS:
        print('saving', m, 'trade history')
        history = get_trade_history(m)
        if history is None or len(history) == 0:
            print('no history!')
            continue

        df = make_history_df(history)
        filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', m) + '.csv.gz'
        if os.path.exists(filename):
            old_df = pd.read_csv(filename, index_col='TimeStamp')
            full_df = old_df.append(df)
            full_df.drop_duplicates(inplace=True)
        else:
            full_df = df

        full_df.to_csv(filename, compression='gzip')

    print('done!\n\n')


def save_all_trade_history_sql():
    """
    saves data to sql
    """
    for m in MARKETS:
        print('saving', m, 'trade history')
        history = get_trade_history(m)
        if history is None or len(history) == 0:
            print('no history!')
            continue

        df = make_history_df(history)
        filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', m) + '.csv.gz'
        if os.path.exists(filename):
            old_df = pd.read_csv(filename, index_col='TimeStamp')
            full_df = old_df.append(df)
            full_df.drop_duplicates(inplace=True)
        else:
            full_df = df

        full_df.to_csv(filename, compression='gzip')

    print('done!\n\n')


def read_history_csv(market):
    filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', market) + '.csv.gz'
    df = pd.read_csv(filename, index_col='TimeStamp')
    return df


def convert_history_to_sql():
    """
    WARNING: will replace all data in SQL tables
    """
    idx = MARKETS.index('BTC-SALT')
    ms = MARKETS[idx:]
    for m in ms:
        print(m)
        engine = create_engine("postgres://nate:{}@localhost:5432/postgres".format(PG_PASS))
        engine.table_names()
        conn = engine.connect()
        conn.execute("commit")
        table_name = '"' + m + '"'
        # try to create db unless already there, then skip creation
        try:
            conn.execute("create database " + table_name + ';')
        except Exception as e:
            print(e)
            pass
        conn.execute("commit")
        conn.close()
        engine = create_engine('postgresql://nate:{}@localhost:5432/{}'.format(PG_PASS, m))
        df = read_history_csv(m)
        df.to_sql(m, engine, if_exists='replace')

    # cursor = conn.cursor()
    # # all_tables = get_all_tables()
    # was starting to do this with psycopg2 but forgot .to_sql in pandas...
    # for m in MARKETS:
    #     table_name = '"' + m + '"'
    #     df = read_history_csv(m)
    #     # create table if doesn't exist
    #     if m not in all_tables:
    #         cursor.execute("""CREATE TABLE {} (
    #                             TradeTime TIMESTAMP,
    #                             FillType VARCHAR,
    #                             Id INTEGER,
    #                             OrderType VARCHAR,
    #                             Price NUMERIC,
    #                             Quantity NUMERIC,
    #                             Total NUMERIC
    #                             );""".format(table_name))
    #     times = pd.to_datetime(df.index).tz_localize('UTC')
    #     row = df.iloc[0]
    #     tup = (m,
    #             times[0],
    #             row['FillType'],
    #             int(row['Id']),
    #             row['OrderType'],
    #             row['Price'],
    #             row['Quantity'],
    #             row['Total'])
    #     cursor.execute("""INSERT INTO %s
    #                         (TradeTime, FillType, Id, OrderType, Price, Quantity, Total)
    #                         VALUES (%s, %s, %s, %s, %s, %s, %s);""", tup)


def get_order_book(market):
    try:
        while True:  # in case of SSL error, keep trying
            try:
                res = requests.get('https://bittrex.com/api/v1.1/public/getorderbook?market=' + market + '&type=both&depth=50000')
                break
            except Exception as e:
                print(e)
                time.sleep(10)

        timestamp = pd.to_datetime(datetime.now())
        if res.json()['success']:
            orders = res.json()['result']
            if orders['buy'] is None and orders['sell'] is None:
                print('error! both buy and sell orders are none')
                return None, None
            return orders, timestamp
        else:
            print('error!', res.json()['message'])
            return None, None
    except Exception as e:
        print(e)
        print('exception! error!')
        return None, None


def make_orderbook_df(orders, timestamp, time_idx=True):
    buy_df = pd.io.json.json_normalize(orders['buy'])
    buy_df['timestamp'] = timestamp
    sell_df = pd.io.json.json_normalize(orders['sell'])
    sell_df['timestamp'] = timestamp

    if time_idx:
        sell_df.set_index('timestamp', inplace=True)
        buy_df.set_index('timestamp', inplace=True)

    return buy_df, sell_df


def make_history_df(history, time_idx=True):
    df = pd.io.json.json_normalize(history)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    if time_idx:
        df.set_index('TimeStamp', inplace=True)

    return df


def make_summary_df(summary):
    keys = summary[0].keys()
    data_dict = {}
    for k in keys:
        data_dict[k] = []

    for s in summary:
        for k in keys:
            data_dict[k].append(s[k])

    df = pd.DataFrame(data_dict)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['Created'] = pd.to_datetime(df['Created'])
    df['24hr_chg'] = df['Last'] - df['PrevDay']
    df['24hr_chg_pct'] = df['24hr_chg'] / df['PrevDay'] * 100
    return df


def save_order_book(market):
    orders, timestamp = get_order_book(market)
    if orders is None and timestamp is None:
        return

    if len(orders['buy']) + len(orders['sell']) == 0:
        print('no orders, skipping')
        return

    buy_df, sell_df = make_orderbook_df(orders, timestamp)
    key = re.sub('-', '_', market)

    buy_file = HOME_DIR + 'data/order_books/buy_orders_' + key + '.csv.gz'
    sell_file = HOME_DIR + 'data/order_books/sell_orders_' + key + '.csv.gz'
    if os.path.exists(buy_file):
        buy_df.to_csv(buy_file, compression='gzip', mode='a', header=False)
        sell_df.to_csv(sell_file, compression='gzip', mode='a', header=False)
    else:
        buy_df.to_csv(buy_file, compression='gzip')
        sell_df.to_csv(sell_file, compression='gzip')

    del orders
    del timestamp
    del buy_df
    del sell_df
    gc.collect()


def save_all_order_books():
    for m in MARKETS:
        print('saving', m, '...')
        save_order_book(m)
        print('sleeping 5s...')
        time.sleep(5)



def read_order_book(market):
    fileend = re.sub('-', '_', market + '.csv.gz')
    buy_df = pd.read_csv(HOME_DIR + 'data/order_books/buy_orders_' + fileend, index_col='timestamp')
    sell_df = pd.read_csv(HOME_DIR + 'data/order_books/sell_orders_' + fileend, index_col='timestamp')
    return buy_df, sell_df


def continuously_save_order_books(interval=600):
    """
    Saves all order books every 'interval' seconds.
    """
    def keep_saving():
        while True:
            save_all_order_books()
            print("\n\ndone.")
            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def continuously_save_trade_history(interval=300):
    """
    Saves all trade history every 'interval' seconds.
    """
    def keep_saving():
        while True:
            save_all_trade_history()
            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def continuously_save_summaries(interval=300):
    """
    Saves all trade history every 'interval' seconds.
    """
    def keep_saving():
        while True:
            save_all_summaries()
            time.sleep(interval)

    thread = Thread(target=keep_saving)
    thread.start()


def get_total_buy_sell_orders():
    """
    Calculates total buy/sell order volume in BTC and USD.
    """
    books = {}
    for m in MARKETS:
        print(m)
        fileend = re.sub('-', '_', m + '.csv.gz')
        if os.path.exists(HOME_DIR + 'data/order_books/buy_orders_' + fileend):
            books[m] = {}
            books[m]['buy'], books[m]['sell'] = read_order_book(m)

    tickers = get_all_tickers()
    ticker_markets = set(tickers.index)
    total_sells = {}
    total_buys = {}
    sells_minus_buys = {}
    cur_pairs = list(books.keys())

    for cur_pair in books.keys():
        if cur_pair not in ticker_markets:
            print('market for', cur_pair, 'not in ticker data')
            continue

        print(cur_pair)
        b = books[cur_pair]
        latest_sell_time = b['sell'].index.unique().max()
        latest_buy_time = b['buy'].index.unique().max()
        latest_sells = b['sell'].loc[latest_sell_time]
        latest_buys = b['buy'].loc[latest_buy_time]
        total_sell = (latest_sells['Quantity'] * tickers.loc[cur_pair]['Last']).sum()
        total_buy = (latest_buys['Quantity'] * latest_buys['Rate']).sum()
        total_sells[cur_pair] = total_sell
        total_buys[cur_pair] = total_buy
        sells_minus_buys[cur_pair] = total_sell - total_buy

    return total_sells, total_buys, sells_minus_buys


def make_buy_sell_df(total_sells, total_buys, sells_minus_buys):
    sells = []
    buys = []
    minus = []
    marks = list(total_sells.keys())
    for m in marks:
        sells.append(total_sells[m])
        buys.append(total_buys[m])
        minus.append(sells_minus_buys[m])

    df = pd.DataFrame({'total_sells':sells,
                        'total_buys':buys,
                        'sells_minus_buys':minus,
                        'MarketName':marks})
    df.set_index('MarketName', inplace=True)
    return df


def get_buy_sell_df():
    total_sells, total_buys, sells_minus_buys = get_total_buy_sell_orders()
    df = make_buy_sell_df(total_sells, total_buys, sells_minus_buys)
    return df
