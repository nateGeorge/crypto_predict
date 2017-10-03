"""
This scrapes data every 10 mins, predicts future prices for all pairs, and
reports which have the biggest changes predicted.

If the most recent training time is more that train_pts away from the current
time (in minutes), then data is scraped fresh and the models re-trained on the
new data.  The initial weights are kept from the previous model.
"""
