Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py"n
    (self.host, self.port), self.timeout, **extra_kw)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/util/connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STR:
  File "/usr/lib/python3.5/socket.py", line 732, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, f:
socket.gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpooln
    chunked=chunked)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpoolt
    self._validate_conn(conn)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpooln
    conn.connect()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py"t
    conn = self._new_conn()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py"n
    self, "Failed to establish a new connection: %s" % e)
urllib3.exceptions.NewConnectionError: <urllib3.connection.VerifiedHTn

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/requests/adapters.py",d
    timeout=timeout
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpooln
    _stacktrace=sys.exc_info()[2])
  File "/usr/local/lib/python3.5/dist-packages/urllib3/util/retry.py"t
    raise MaxRetryError(_pool, url, error or ResponseError(cause))
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='bittrex.c)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.5/threading.py", line 914, in _bootstrap_innr
    self.run()
  File "/usr/lib/python3.5/threading.py", line 862, in run
    self._target(*self._args, **self._kwargs)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py",g
    save_all_trade_history()
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py",y
    history = get_trade_history(m)
  File "/media/nate/data_lake/crytpo_predict/code/scrape_bittrex.py",y
    res = requests.get('https://bittrex.com/api/v1.1/public/getmarket)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", linet
    return request('get', url, params=params, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", linet
    return session.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py",t
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py",d
    r = adapter.send(request, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/adapters.py",d
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='bittre)
