import os
import requests
import time
from datetime import datetime
import pandas as pd
import re
from threading import Thread


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
    res = requests.get('https://bittrex.com/api/v1.1/public/getmarkets')
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


HOME_DIR = get_home_dir()
MARKETS = get_all_currency_pairs()


def get_all_summaries():
    res = requests.get('https://bittrex.com/api/v1.1/public/getmarketsummaries')
    if res.json()['success']:
        summary = res.json()['result']
        return summary
    else:
        print('error! ', res.json()['message'])
        return None


def get_all_tickers():
    tickers = []
    for m in MARKETS:
        res = requests.get('https://bittrex.com/api/v1.1/public/getticker?market=' + m)
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
    res = requests.get('https://bittrex.com/api/v1.1/public/getmarkethistory?market=' + market)
    try:
        if res.json()['success']:
            history = res.json()['result']
            return history
        else:
            print('error! ', res.json()['message'])
            return None
    except:
        print('exception! error!')
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


def read_history(market):
    filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', market) + '.csv.gz'
    df = pd.read_csv(filename, index_col='TimeStamp')
    return df


def get_order_book(market):
    try:
        # was having some weird error here, so moved the requests.get into the try block
        res = requests.get('https://bittrex.com/api/v1.1/public/getorderbook?market=' + market + '&type=both&depth=50000')
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
    except:
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


def save_all_order_books():
    for m in MARKETS:
        print('saving', m, '...')
        save_order_book(m)


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
