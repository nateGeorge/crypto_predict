# designed for python 3
# meant to be run from the home repo directory

import quandl
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import glob
import bittrex_test as btt
# only required for >50 calls per day
key = os.environ.get('QUANDL_KEY')
if key == None:
    key = os.environ.get('quandl_api')

if key == None:
    print('warning! no quandl api key found in bashrc')

quandl.ApiConfig.api_key = key

HOME_DIR = btt.get_home_dir()

# vcx doesn't seem reputable.  0% exchange fees, and their SSL cert
# has been expired 3 months

def show_all_choices():
    # rock looks like it is a reputable, and has been around since 2011 to present
    market_data['rock']['Close'].plot()
    plt.show()
    # holes in the data...

    # less bad...
    market_data['btce']['Close'].plot()
    plt.show()

    # super holy
    market_data['vcx']['Close'].plot()
    plt.show()

    # looks best!
    market_data['bitstamp']['Close'].plot()
    plt.show()


def get_markets():
    usd_markets = pd.read_csv(HOME_DIR + 'data/usd_markets_quandl.csv', header=None).values.tolist()[0]
    market_data = {}
    bad_markets = set(['bcmML', 'fresh', 'ripple'])
    to_get = set(usd_markets) - bad_markets
    return to_get


def dl_all_data():
    to_get = get_markets()
    market_data = {}
    for m in to_get:
        print('downloading', m + '...')
        market_data[m] = quandl.get('BCHARTS/' + m.upper() + 'USD')

    return market_data


def download_data(path=HOME_DIR + 'data/bitcoin_prices/', check_latest=False, save=True):
    usd_markets = pd.read_csv(HOME_DIR + 'data/usd_markets_quandl.csv', header=None).values.tolist()[0]
    market_data = {}
    bad_markets = set(['bcmML', 'fresh', 'ripple'])
    if check_latest:
        for m in usd_markets:
            # problem with bcmML for some reason
            if m in bad_markets:
                continue
            print('BCHARTS/' + m.upper() + 'USD')
            market_data[m] = quandl.get('BCHARTS/' + m.upper() + 'USD')
        # usd_btc = quandl.get('BCHARTS/anxhkUSD')
    # checking which markets have 2017 data
        up_to_date = []
        for m in market_data.keys():
            if 2017==market_data[m].index.max().year:
                print(m)
                up_to_date.append(m)

        if save:
            for m in up_to_date:
                print('saving', m, '...')
                market_data[m].to_csv(path.strip('/') + '/' + m)
                print(market_data[m].index.min())

    else:
        to_get = ['bitkonan', 'bitstamp', 'btce', 'localbtc', 'rock', 'vcx']
        for m in to_get:
            market_data[m] = quandl.get('BCHARTS/' + m.upper() + 'USD')
            if save:
                print('saving', m, '...')
                market_data[m].to_csv(path.strip('/') + '/' + m)

    return market_data


def load_save_data_old(clear_others=True):
    """
    If it doesn't exist, saves the current data in a folder with today's date.
    Otherwise, downloads data and saves it.

    clear_others will delete the other folders from the past
    """
    today = datetime.datetime.now().strftime('%Y-%m-%d') # year-month-day
    if not os.path.exists(today):
        print('scraping fresh...')
        os.mkdir(today)
        market_data = download_data(today)
    else:
        print('up to date on a daily basis...loading...')
        market_data = {}
        file_list = glob.iglob(today + '/*')
        # thought I needed to get just filename, but wrong
        # file_list = [s.split('/')[1] for s in file_list]
        for l in file_list:
            print(l)
            market_data[l.split('/')[1]] = pd.read_csv(l)

    return market_data


def load_save_data():
    today = datetime.datetime.now().strftime('%Y-%m-%d') # year-month-day
    data_path = HOME_DIR + 'data/bitcoin_prices/'
    comp_path = data_path + 'composite.csv'
    bitstamp_path = data_path + 'bitstamp.csv'
    to_get = get_markets()
    if os.path.exists(bitstamp_path):
        bitstamp_df = pd.read_csv(bitstamp_path,
                                    index_col='Date',
                                    parse_dates=True)
        latest_date = bitstamp_df.index.max()
        latest_bitstamp = quandl.get('BCHARTS/BITSTAMPUSD')
        if latest_bitstamp.index.max() == latest_date:
            print('up to date on a daily basis...loading...')
            market_data = {}
            for m in to_get:
                market_data[m] = pd.read_csv(data_path + m + '.csv',
                                            index_col='Date', parse_dates=True)
        else:
            print('scraping fresh...')
            market_data = dl_all_data()
            for m in to_get:
                market_data[m].to_csv(data_path + m + '.csv')
    else:
        print('scraping fresh...')
        market_data = dl_all_data()
        for m in to_get:
            market_data[m].to_csv(data_path + m + '.csv')
        # make composite price index and save


    return market_data


def make_composite_idx(market_data):
    """
    Averages prices, taking weighted average of price based on volumes.
    Work in progress
    """
    all_dates = []
    for m in market_data.keys():
        all_dates.extend(market_data[m].index.values)

    all_dates = np.array(all_dates)
    min_date = min(all_dates)
    max_date = max(all_dates)
    dt_range = pd.date_range(start=min_date, end=max_date)  # default freq is day
    # todo: finish function


def get_bitstamp_full_df(market_data):
    bitstamp_df = market_data['bitstamp']
    btce_df = market_data['btce']
    missing_dates = bitstamp_df[bitstamp_df['Open'] == 0].index
    cols = bitstamp_df.columns.tolist()
    for d in missing_dates:
        for c in cols:
            bitstamp_df.set_value(d, c, btce_df.loc[d, c])

    # bitstamp_df['Weighted Price'].plot()
    # plt.show()

    return bitstamp_df


if __name__=="__main__":
    market_data = load_save_data()
    # getting places where data is 0 and correcting


    #market_data['bitstamp']['Open'][market_data['bitstamp']['Open'] == 0].index
    # btce is the only one that has the missing data
