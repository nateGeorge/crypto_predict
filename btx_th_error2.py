saving BTC-CANN trade history
saving BTC-SYS trade history
Exception in thread Thread-122:
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/contrib/pyopenssl.py", line 441,
in wrap_socket
    cnx.do_handshake()
  File "/usr/local/lib/python3.5/dist-packages/OpenSSL/SSL.py", line 1638, in do_handsha
ke
    self._raise_ssl_error(self._ssl, result)
  File "/usr/local/lib/python3.5/dist-packages/OpenSSL/SSL.py", line 1370, in _raise_ssl
_error
    raise SysCallError(errno, errorcode.get(errno))
OpenSSL.SSL.SysCallError: (104, 'ECONNRESET')

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 601, in
urlopen
    chunked=chunked)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 346, in
_make_request
    self._validate_conn(conn)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connectionpool.py", line 850, in
_validate_conn
    conn.connect()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/connection.py", line 326, in conn
ect
    ssl_context=context)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/util/ssl_.py", line 329, in ssl_w
rap_socket
    return context.wrap_socket(sock, server_hostname=server_hostname)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/contrib/pyopenssl.py", line 448, in wrap_socket
    raise ssl.SSLError('bad handshake: %r' % e)
ssl.SSLError: ("bad handshake: SysCallError(104, 'ECONNRESET')",)


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
  File "/usr/local/lib/python3.5/dist-packages/requests/adapters.py", line 506, in send
    raise SSLError(e, request=request)
requests.exceptions.SSLError: HTTPSConnectionPool(host='bittrex.com', port=443): Max retries exceeded with url: /api/v1.1/public/getmarkethistory?market=BTC-SYS (Caused by SSLError(SSLError("bad handshake: SysCallError(104, 'ECONNRESET')",),))
