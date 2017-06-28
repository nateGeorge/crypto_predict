# crytpo_predict
Predicts price trends for cryptocurrencies.

# get started
Install requirements:

`pip install -r requirements.txt`

(I have to do `sudo pip3 install -r requirements.txt`)

Also need to install async wrapper from here: https://github.com/absortium/poloniex-api
no pip package as of now.

# todo
* put try: except in continuous scrapes in scrape_bittrex.py
* try to debug error if it happens again in continuous scrapes
* port neural net model to data here
* look for order book correlations to price movement
* neural net with order book and price/vol history
