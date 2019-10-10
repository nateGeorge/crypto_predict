# crytpo_predict
Predicts price trends for cryptocurrencies.  This repo is pretty messy.  It also scrapes bittrex and poloniex data.

# get started
Install requirements:

`pip install -r requirements.txt`

(I have to do `sudo pip3 install -r requirements.txt`)

Also need to install async wrapper from here: https://github.com/absortium/poloniex-api
no pip package as of now.

# making the PSQL database
First make sure postgresql is installed.  Then do
`sudo su postgres`
to enter a shell as the postgres user.  Then do
`psql`
to enter the postgresql interface.
Next create a user:
`CREATE USER nate WITH SUPERUSER CREATEDB PASSWORD 'testing123';`
https://stackoverflow.com/a/15008311/4549682
https://www.postgresql.org/docs/9.5/static/sql-createuser.html
If you don't get the permissions fully correct, you can go back and alter the user:
https://www.postgresql.org/docs/9.5/static/sql-alteruser.html
The semicolons at the end are required; otherwise it won't work.

## accessing postgresql database
`sudo -u postgres psql postgres`
or
`sudo -u postgres psql nate`
(username is last string)
`\l` lists tables
`\c bittrex` connects to bittrex database
`SELECT * from bittrex LIMIT 1;` -- gets one entry
`SELECT count(*) from bittrex;` -- gets number of entries in table

# daily scraping
## Poloniex
Probably best not to use cron actually, because when the computer shuts down, the files can get corrupted.
Instead, should write a script that will start the scraping in tmux, so it can be stopped in a controlled fashion before rebooting.

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
