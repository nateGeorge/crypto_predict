saving BTC-GEO trade history
saving BTC-FLDC trade history
Exception in thread Thread-88:
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py", line 141, in _new_conn
    (self.host, self.port), self.timeout, **extra_kw)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/util/connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
  File "/usr/lib/python3.5/socket.py", line 732, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 601, in urlopen
    chunked=chunked)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 346, in _make_request
    self._validate_conn(conn)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 850, in _validate_conn
    conn.connect()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py", line 284, in connect
    conn = self._new_conn()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py", line 150, in _new_conn
    self, "Failed to establish a new connection: %s" % e)
urllib3.exceptions.NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x7f22c45c4898>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/requests/adapters.py", line 440, in send
    timeout=timeout
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 639, in urlopen
    _stacktrace=sys.exc_info()[2])
  File "/usr/local/lib/python3.5/dist-packages/urllib3/util/retry.py", line 388, in increment
    raise MaxRetryError(_pool, url, error or ResponseError(cause))
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='bittrex.com', port=443): Max retries exceeded with url: /api/v1.1/public/getmarkethistory?market=BTC-FLDC (Caused by NewConnectionError('<urllib3.conne)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.5/threading.py", line 914, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.5/threading.py", line 862, in run
    self._target(*self._args, **self._kwargs)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 225, in keep_saving
    save_all_trade_history()
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 89, in save_all_trade_history
    history = get_trade_history(m)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py", line 73, in get_trade_history
    res = requests.get('https://bittrex.com/api/v1.1/public/getmarkethistory?market=' + market)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 72, in get
    return request('get', url, params=params, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 58, in request
    return session.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 508, in request
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 618, in send
    r = adapter.send(request, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/adapters.py", line 508, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='bittrex.com', port=443): Max retries exceeded with url: /api/v1.1/public/getmarkethistory?market=BTC-FLDC (Caused by NewConnectionError('<urllib3.co)
