# crytpo_predict
Predicts price trends for cryptocurrencies.

# get started
Install requirements:

`pip install -r requirements.txt`

(I have to do `sudo pip3 install -r requirements.txt`)

Also need to install async wrapper from here: https://github.com/absortium/poloniex-api
no pip package as of now.

# daily scraping
## Poloniex
To scrape poloniex daily, add this to crontab (`crontab -e`):
(min hr day month weekday file)

`*/10 * * * * /usr/bin/python3 /home/nate/github/crypto_predict_latest/crypto_predict/code/poloniex/scrape_polo.py >> /home/nate/github/crypto_predict_latest/crypto_predict/polo_scrape_log.log`

This will run every 10 minutes.  Be sure to change the home directory if needed.


# todo
* put try: except in continuous scrapes in scrape_bittrex.py
* try to debug error if it happens again in continuous scrapes
* port neural net model to data here
* look for order book correlations to price movement
* neural net with order book and price/vol history

# installing TA-Lib
first you need to install ta-lib: https://sourceforge.net/projects/ta-lib/files/ta-lib/
may need some dependencies on Ubuntu: https://stackoverflow.com/questions/26053982/error-setup-script-exited-with-error-command-x86-64-linux-gnu-gcc-failed-wit
