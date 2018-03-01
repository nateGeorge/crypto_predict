saving ETH-PAY trade history
exception! error!
Exception in thread Thread-103:
Traceback (most recent call last):
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 86, in get_trade_history
    if res.json()['success']:
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 892, in json
    return complexjson.loads(self.text, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/simplejson/__init__.py", line 516, in loads
    return _default_decoder.decode(s)
  File "/usr/local/lib/python3.5/dist-packages/simplejson/decoder.py", line 374, in decode
    obj, end = self.raw_decode(s)
  File "/usr/local/lib/python3.5/dist-packages/simplejson/decoder.py", line 404, in raw_decode
    return self.scan_once(s, idx=_w(s, idx).end())
simplejson.scanner.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.5/threading.py", line 914, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.5/threading.py", line 862, in run
    self._target(*self._args, **self._kwargs)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 241, in keep_saving
    save_all_trade_history()
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 101, in save_all_trade_history
    history = get_trade_history(m)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 94, in get_trade_history
    print(e.message, e.args)
AttributeError: 'JSONDecodeError' object has no attribute 'message'
