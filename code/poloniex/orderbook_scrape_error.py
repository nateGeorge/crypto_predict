Exception in thread Thread-168:
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 244, in handleReturned
    out = _loads(data, parse_float=str)
  File "/usr/lib/python3.5/json/__init__.py", line 332, in loads
    return cls(**kw).decode(s)
  File "/usr/lib/python3.5/json/decoder.py", line 339, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/usr/lib/python3.5/json/decoder.py", line 357, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.5/threading.py", line 914, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.5/threading.py", line 862, in run
    self._target(*self._args, **self._kwargs)
  File "/home/nate/github/crypto_predict_latest/crypto_predict/code/poloniex/scrape_polo.py", line 113, in keep_saving
    save_all_order_books()
  File "/home/nate/github/crypto_predict_latest/crypto_predict/code/poloniex/scrape_polo.py", line 101, in save_all_order_books
    buy_dfs, sell_dfs = get_all_orderbooks()
  File "/home/nate/github/crypto_predict_latest/crypto_predict/code/poloniex/scrape_polo.py", line 56, in get_all_orderbooks
    orderbooks = polo.returnOrderBook(currencyPair='all', depth=1000000)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 290, in returnOrderBook
    'depth': str(depth)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 143, in retrying
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 218, in __call__
    return self.handleReturned(ret.text)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 251, in handleReturned
    raise PoloniexError('Invalid json response returned')
poloniex.PoloniexError: Invalid json response returned

