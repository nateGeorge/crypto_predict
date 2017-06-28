from selenium import webdriver
import time
import datetime
import pandas as pd

# TODO: save data and don't scrape fresh unless not the same day as last scraped
# or forced to rescrape

def scrape_data(currency='ethereum'):
    driver = webdriver.PhantomJS()
    driver.get('https://coinmarketcap.com/currencies/' + currency + '/')
    chart_number = '0'  # always 0 for now
    # series 0 is market cap
    # series 1 is USD price
    # series 2 is BTC price
    # series 3 is volume
    # series 4? not sure yet.  min 0 though, max 35607135070, maybe btc market cap
    # x-axis in unix timestamp, milliseconds since epoch UTC

    # if btc,
    usd_data = driver.execute_script('return Highcharts.charts[' + chart_number + '].series[1].options.data')
    usd_price = []
    udates = []
    for point in usd_data:
        usd_price.append(point[1])
        udates.append(datetime.datetime.utcfromtimestamp(point[0] / 1000.))

    vol_data = driver.execute_script('return Highcharts.charts[' + chart_number + '].series[3].options.data')
    volume = []
    vdates = []
    for point in vol_data:
        volume.append(point[1])
        vdates.append(datetime.datetime.utcfromtimestamp(point[0] / 1000.))

    # test to make sure dates are the same before putting in dataframe
    for u, v in zip(udates, vdates):
        if u != v:
            print('uhoh! datapoint doesn\'t match up in dates')
            print(u, v)

    df = pd.DataFrame({'date':udates, 'usd_price':usd_price, 'volume':volume})
    df.set_index('date', inplace=True)
    driver.quit()
    return df


if __name__ == "__main__":
    # driver = webdriver.Chrome()  # only for debugging--brings up actual browser
    eth = scrape_data()
    btc = scrape_data('bitcoin')
    aeon = scrape_data('aeon')
    xmr = scrape_data('monero')
