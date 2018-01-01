scraping updates
saving BTC_AMP
checking BTC_ARDR
scraping updates
saving BTC_ARDR
checking BTC_BCH
scraping updates
scraping another time...
('Connection broken: IncompleteRead(0 bytes read)', IncompleteRead(0 bytes read))
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 543, in _updat
e_chunk_length
    self.chunk_left = int(line, 16)
ValueError: invalid literal for int() with base 16: b''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 302, in _error
_catcher
    yield
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 598, in read_c
hunked
    self._update_chunk_length()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 547, in _upda$
e_chunk_length
    raise httplib.IncompleteRead(line)
http.client.IncompleteRead: IncompleteRead(0 bytes read)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 745, in generat
e
    for chunk in self.raw.stream(chunk_size, decode_content=True):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 432, in stream
    for line in self.read_chunked(amt, decode_content=decode_content):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 626, in read_c
hunked
    self._original_response.close()
  File "/usr/lib/python3.5/contextlib.py", line 77, in __exit__
    self.gen.throw(type, value, traceback)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 320, in _error
_catcher
    raise ProtocolError('Connection broken: %r' % e, e)
urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead(0 bytes read)', In
completeRead(0 bytes read))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 143, in retry
ing
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 308, in marke
tTradeHist
    timeout=self.timeout)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 72, in get
    return request('get', url, params=params, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 58, in request
    return session.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 508, in reque
st
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 658, in send
    r.content
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 823, in content
    self._content = bytes().join(self.iter_content(CONTENT_CHUNK_SIZE)) or bytes()
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 748, in generat
e
    raise ChunkedEncodingError(e)
requests.exceptions.ChunkedEncodingError: ('Connection broken: IncompleteRead(0 bytes re
ad)', IncompleteRead(0 bytes read))
scraping another time...
('Connection broken: IncompleteRead(0 bytes read)', IncompleteRead(0 bytes read))
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 543, in _updat
e_chunk_length
    self.chunk_left = int(line, 16)
ValueError: invalid literal for int() with base 16: b''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 302, in _error
_catcher
    yield
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 598, in read_c
hunked
    self._update_chunk_length()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 547, in _updat
e_chunk_length
    raise httplib.IncompleteRead(line)
http.client.IncompleteRead: IncompleteRead(0 bytes read)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 745, in generat
e
    for chunk in self.raw.stream(chunk_size, decode_content=True):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 432, in stream
    for line in self.read_chunked(amt, decode_content=decode_content):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 626, in read_c
hunked
    self._original_response.close()
  File "/usr/lib/python3.5/contextlib.py", line 77, in __exit__
    self.gen.throw(type, value, traceback)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 320, in _erro$
_catcher
    raise ProtocolError('Connection broken: %r' % e, e)
urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead(0 bytes read)', I$
completeRead(0 bytes read))


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 143, in retry
ing
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 308, in marke
tTradeHist
    timeout=self.timeout)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 72, in get
    return request('get', url, params=params, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 58, in request
    return session.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 508, in reque
st
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 658, in send
    r.content
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 823, in content
    self._content = bytes().join(self.iter_content(CONTENT_CHUNK_SIZE)) or bytes()
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 748, in generat
e
    raise ChunkedEncodingError(e)
requests.exceptions.ChunkedEncodingError: ('Connection broken: IncompleteRead(0 bytes re
ad)', IncompleteRead(0 bytes read))
scraping another time...
scraping another time...
('Connection broken: IncompleteRead(0 bytes read)', IncompleteRead(0 bytes read))
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 543, in _updat
e_chunk_length
    self.chunk_left = int(line, 16)
ValueError: invalid literal for int() with base 16: b''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 302, in _error
_catcher
    yield
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 598, in read_$hunked
    self._update_chunk_length()
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 547, in _upda$e_chunk_length
    raise httplib.IncompleteRead(line)
http.client.IncompleteRead: IncompleteRead(0 bytes read)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 745, in generate
    for chunk in self.raw.stream(chunk_size, decode_content=True):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 432, in stream
    for line in self.read_chunked(amt, decode_content=decode_content):
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 626, in read_chunked
    self._original_response.close()
  File "/usr/lib/python3.5/contextlib.py", line 77, in __exit__
    self.gen.throw(type, value, traceback)
  File "/usr/local/lib/python3.5/dist-packages/urllib3/response.py", line 320, in _error_catcher
    raise ProtocolError('Connection broken: %r' % e, e)
urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead(0 bytes read)', IncompleteRead(0 bytes read))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 143, in retrying
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/poloniex/__init__.py", line 308, in marketTradeHist
    timeout=self.timeout)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 72, in get
    return request('get', url, params=params, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/api.py", line 58, in request
    return session.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 508, in requ$st
    resp = self.send(prep, **send_kwargs)
  File "/usr/local/lib/python3.5/dist-packages/requests/sessions.py", line 658, in send
    r.content
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 823, in conten$
    self._content = bytes().join(self.iter_content(CONTENT_CHUNK_SIZE)) or bytes()
  File "/usr/local/lib/python3.5/dist-packages/requests/models.py", line 748, in genera$e
    raise ChunkedEncodingError(e)
requests.exceptions.ChunkedEncodingError: ('Connection broken: IncompleteRead(0 bytes r$ad)', IncompleteRead(0 bytes read))
