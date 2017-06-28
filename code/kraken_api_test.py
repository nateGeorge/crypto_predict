# requires python 3, because of krakenex package:
# https://github.com/veox/python3-krakenex

import krakenex as kkx
import os

key = os.environ.get('KRAKEN_KEY')
secret = os.environ.get('KRAKEN_SECRET')

kr_api = kkx.API(key=key, secret=secret)

def get_trading_pairs():
    """
    queries tradable asset pairs
    seems to follow the convention of returning a dict with keys: error, result
    within 'result' is a dict of asset pairs as keys, following the convention
    Xsymb1Zsymb2 if symb2 is a regular currency, otherwise Xsymb1Xsymb2
    if symb2 is a cryptocurrency
    """
    pairs = kr_api.query_public('AssetPairs')
    if pairs['error'] == []:
        print('great success!')
        pair_keys = [p for p in pairs['result'].keys()]
    return pairs, pair_keys

"""
defaults to 1 min interval
returns dict with keys ['error', 'result']
data is in ohlc_data['result']['XXBTZUSD'] as
entries(<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>)
appears only to return last 710 minutes (about 12 hours)
"""
q_dict = {'pair': 'XXBTZUSD', 'interval':1, 'since':1489803600}
ohlc_data = kr_api.query_public('OHLC', req=q_dict)
