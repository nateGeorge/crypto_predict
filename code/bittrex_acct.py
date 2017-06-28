from bittrex import Bittrex
import requests
import pandas as pd
import os
import bittrex_test as btt
import quandl_api_test as qat
from scrape_coinmarketcap import scrape_data

API_K = os.environ.get('bittrex_api')
API_S = os.environ.get('bittrex_sec')
if API_K is None:
    API_K = os.environ.get('btx_key')
    API_S = os.environ.get('btx_sec')

bt = Bittrex(API_K, API_S)

HOME_DIR = btt.get_home_dir()
MARKETS = btt.get_all_currency_pairs()


def get_balances():
    bals = bt.get_balances()
    if bals['success'] == True:
        return pd.io.json.json_normalize(bals['result'])
    else:
        print('error!', bals['message'])
        return None


def get_total_dollar_balance(bals):
    btc_amts = []
    dollar_amts = []
    for i, r in bals.iterrows():
        if 'BTC-' + r['Currency'] in MARKETS:
            print('getting price for', r['Currency'])
            t = btt.get_ticker('BTC-' + r['Currency'])
            btc_amts.append(t['Last'] * r['Balance'])
        else:
            # have to find which market we have, the convert to BTC
            if r['Currency'] == 'BTC':
                btc_amts.append(r['Balance'])
            else:
                print('no BTC market for', r['Currency'])

    bals_copy = bals.copy()
    bals_copy['BTC_equivalent'] = btc_amts
    usdt = btt.get_ticker('USDT-BTC')['Last']
    bals_copy['USD_equivalent'] = bals_copy['BTC_equivalent'] * usdt

    return bals_copy


def get_deposit_history():
    dh = bt.get_deposit_history()
    if dh['success'] == True:
        df = pd.io.json.json_normalize(dh['result'])
        df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
        return df
    else:
        print('error!', dh['message'])
        return None


def get_deposit_amts(df):
    # market_data = qat.load_save_data()
    # bt_df = qat.get_bitstamp_full_df(market_data)
    eth = scrape_data()
    btc = scrape_data('bitcoin')
    aeon = scrape_data('aeon')
    xmr = scrape_data('monero')
    dep_dollars = []
    for i, r in df.iterrows():
        date = r['LastUpdated']
        d = str(date.day).zfill(2)
        m = str(date.month).zfill(2)
        y = str(date.year)
        if r['Currency'] == 'BTC':
            price = btc.loc[y + m + d, 'usd_price'][0]
            dep_dollars.append(price * r['Amount'])
        elif r['Currency'] == 'ETH':
            price = eth.loc[y + m + d, 'usd_price'][0]
            dep_dollars.append(price * r['Amount'])
        elif r['Currency'] == 'AEON':
            price = aeon.loc[y + m + d, 'usd_price'][0]
            dep_dollars.append(price * r['Amount'])
        elif r['Currency'] == 'XMR':
            price = xmr.loc[y + m + d, 'usd_price'][0]
            dep_dollars.append(price * r['Amount'])

    df['usd'] = dep_dollars
    return df


def get_order_history():
    hist = bt.get_order_history()
    if hist['success']:
        df = pd.io.json.json_normalize(hist['result'])
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df.sort_values(by='TimeStamp', inplace=True)
        return df
    else:
        print('error!', hist['message'])
        return None


def get_pct_gain_by_coin():
    print('getting balances...')
    bals = get_balances()
    print('done.')
    print('getting total $ bal')
    bals = get_total_dollar_balance(bals)
    print('done')
    print('getting order history')
    hist = get_order_history()
    print('done')
    total_gain_loss = 0
    currencies = hist['Exchange'].unique()
    for c in currencies:
        if c[:3] != 'BTC':
            print('uh-oh, this one isn\'t traded for BTC:', c)
            break

        trades = hist[hist['Exchange'] == c]
        sells = trades[trades['OrderType'].str.contains('SELL')]
        buys = trades[trades['OrderType'].str.contains('BUY')]
        if sells['Quantity'].sum() == buys['Quantity'].sum():
            sold = (sells['PricePerUnit'] * sells['Quantity']).sum()  # in BTC
            bought = (buys['PricePerUnit'] * buys['Quantity']).sum()
            gain_loss = sold - bought
            print('gain/loss from', c, ':', gain_loss, 'BTC')
        else:
            sold_qty = sells['Quantity'].sum()
            sold_left = sold_qty
            sold = (sells['Quantity'] * sells['PricePerUnit']).sum()
            bought_qty = 0
            bought = 0
            for i, b in buys.iterrows():
                bought_qty += b['Quantity']
                if bought_qty > sold_qty:
                    bought = sold_qty * b['PricePerUnit']
                    break
                else:
                    bought_qty += b['Quantity']
                    sold_left -= b['Quantity']
                    bought += (b['Quantity'] * b['PricePerUnit'])

            gain_loss = sold-bought
            print('gain/loss from', c, ':', gain_loss, 'BTC')
            print('still have', buys['Quantity'].sum() - sold_qty)

        total_gain_loss += gain_loss

    print('total gain/loss:', total_gain_loss, 'BTC')


if __name__ == "__main__":
    bals = get_balances()
    df = get_deposit_history()

    # bals = get_total_dollar_balance(bals)
