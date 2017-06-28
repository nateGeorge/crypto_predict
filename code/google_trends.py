import pandas as pd
import bittrex_test as btt
import datetime
import quandl_api_test as qat

HOME_DIR = btt.get_home_dir()

bitcoin_trend = pd.read_csv(HOME_DIR + 'data/google_trends/btc_6-8-2017.csv',
                            skiprows=1,
                            index_col='Week',
                            parse_dates=True)

market_data = qat.load_save_data()
bitstamp_df = qat.get_bitstamp_full_df(market_data)

# create weekly bitcoin price averages
rs_bitstamp = bitstamp_df.loc[bitcoin_trend.index.min() - pd.Timedelta(days=7):]
# need to do some manual calculations to see how this is actually working...
rs_bitstamp2 = rs_bitstamp.resample('W-SUN', label='right').mean()
rs_bitstamp2 = rs_bitstamp2.loc[bitcoin_trend.index.min():bitcoin_trend.index.max()]

# plots correlation of all data
plt.scatter(rs_bitstamp2['Weighted Price'], bitcoin_trend['bitcoin: (Worldwide)'])
plt.xlabel('bitstamp price')
plt.ylabel('bitcoin search trend value')
plt.show()

# plots just up to 2016
bs_2016 = rs_bitstamp2.loc[:'2016-01-01']
gt_2016 = bitcoin_trend.loc[:'2016-01-01']
plt.scatter(bs_2016['Weighted Price'], gt_2016['bitcoin: (Worldwide)'])
plt.xlabel('bitstamp price')
plt.ylabel('bitcoin search trend value')
plt.show()

# plots both lines
bs_2016['Weighted Price'].plot(label='bitstamp weighted price', color='blue')
ax1 = plt.gca()
ax1.set_ylabel('bitstamp price')
ax2 = ax1.twinx()
gt_2016['bitcoin: (Worldwide)'].plot(ax=ax2, color='orange', label='google trend')
ax2.set_ylabel('google trend for bitcoin')
plt.legend()
plt.show()

# plots both lines for full dataset
rs_bitstamp2['Weighted Price'].plot(label='bitstamp weighted price', color='blue')
ax1 = plt.gca()
ax1.set_ylabel('bitstamp price')
ax2 = ax1.twinx()
bitcoin_trend['bitcoin: (Worldwide)'].plot(ax=ax2, color='orange', label='google trend')
ax2.set_ylabel('google trend for bitcoin')
plt.legend()
plt.show()
