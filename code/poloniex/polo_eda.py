import pandas as pd
from glob import iglob
import os


def get_home_dir(repo_name='crypto_predict'):
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo_name]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir


HOME_DIR = get_home_dir()


def read_orderbook(market='BTC_AMP'):
    datapath = HOME_DIR + 'data/order_books/poloniex/'
    buy_file = datapath + 'buy_orders_' + market + '.csv.gz'
    sell_file = datapath + 'sell_orders_' + market + '.csv.gz'
    bdf = pd.read_csv(buy_file, index_col='timestamp')
    sdf = pd.read_csv(sell_file, index_col='timestamp')
    return bdf, sdf


def read_trade_hist(market='BTC_AMP', drop=0):
    datapath = HOME_DIR + 'data/trade_history/poloniex/'
    filename = datapath + market + '.csv.gz'
    df = pd.read_csv(filename, index_col='date', parse_dates=True)
    # sometimes might want to drop the first few thousand points because
    # they are crazy
    df = df.iloc[drop:]
    return df

def get_all_trade_pairs():
    datapath = HOME_DIR + 'data/trade_history/poloniex/'
    pairs = [f.split('/')[-1][14:-7] for f in iglob(datapath + '*.csv.gz')]
    return pairs
