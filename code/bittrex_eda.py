import requests
import pandas as pd
from datetime import datetime
import re
import os
import matplotlib.pyplot as plt
import bittrex_test as btt
import bittrex_acct as ba
from bittrex import Bittrex

HOME_DIR = btt.get_home_dir()
MARKETS = btt.get_all_currency_pairs()


def read_order_book(market):
    fileend = re.sub('-', '_', market + '.csv.gz')
    buy_df = pd.read_csv(HOME_DIR + 'data/order_books/buy_orders_' + fileend, index_col='timestamp')
    sell_df = pd.read_csv(HOME_DIR + 'data/order_books/sell_orders_' + fileend, index_col='timestamp')
    return buy_df, sell_df


def read_trade_history(market):
    filename = HOME_DIR + 'data/trade_history/' + re.sub('-', '_', market) + '.csv.gz'
    df = pd.read_csv(filename, index_col='TimeStamp')
    return df


def plot_current_orderbook(bdf, sdf, market=None):
    times = bdf.index.unique().tolist()
    times2 = sdf.index.unique().tolist()
    latest_b = bdf.loc[times[-1]]
    latest_s = sdf.loc[times[-1]]
    latest_b = latest_b.sort_values(by='Rate')
    latest_b['cum_quant'] = latest_b['Quantity'].iloc[::-1].cumsum().iloc[::-1]
    # this looks strange but maybe offers some insights?
    latest_b['weird_cum_rate'] = latest_b['cum_quant'] * latest_b['Rate']
    latest_b['btc_bids'] = latest_b['Quantity'] * latest_b['Rate']
    latest_b['cum_btc'] = latest_b['btc_bids'].iloc[::-1].cumsum().iloc[::-1]

    latest_s = latest_s.sort_values(by='Rate')
    latest_s['cum_quant'] = latest_s['Quantity'].cumsum()
    latest_s['weird_cum_rate'] = latest_s['cum_quant'] * latest_s['Rate']
    latest_s['btc_bids'] = latest_s['Quantity'] * latest_s['Rate']
    latest_s['cum_btc'] = latest_s['btc_bids'].cumsum()

    # latest_b.plot(x='Rate', y='cum_btc', label='buy')
    latest_b.plot(x='Rate', y='cum_quant', label='buy')
    ax = plt.gca()
    # latest_s.plot(x='Rate', y='cum_btc', ax=ax, label='sell')
    latest_s.plot(x='Rate', y='cum_quant', ax=ax, label='sell')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if market is not None:
        ax.set_title(market)
    plt.show()


def plot_all_obs(bals):
    for c in bals.loc[20:].Currency:
        if c == 'BTC':
            market = 'USDT-BTC'
        elif c == 'QRL':
            continue
        else:
            market = 'BTC-' + c

        bdf, sdf = btt.read_order_book(market)
        plot_current_orderbook(bdf, sdf, market)


if __name__ == "__main__":
    market = 'BTC-UBQ'
    market = 'BTC-STEEM'
    market = 'USDT-BTC'
    bals = ba.get_balances()
