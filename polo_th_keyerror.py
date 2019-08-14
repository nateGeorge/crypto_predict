Traceback (most recent call last):
  File "/media/nate/nates/github/crypto_predict/code/poloniex/scrape_polo.py", line 712, in keep_saving
    save_all_trade_history_psql()
  File "/media/nate/nates/github/crypto_predict/code/poloniex/scrape_polo.py", line 668, in save_all_trade_history_psql
    df, update = get_trade_history(c, two_h_delay=two_h_delay)
  File "/media/nate/nates/github/crypto_predict/code/poloniex/scrape_polo.py", line 575, in get_trade_history
    full_df.sort_values(by='globaltradeid', inplace=True)
  File "/home/nate/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py", line 4719, in sort_values
    k = self._get_label_or_level_values(by, axis=axis)
  File "/home/nate/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py", line 1707, in _get_label_or_level_values
    raise KeyError(key)
KeyError: 'globaltradeid'
