from poloniex.app import AsyncApp
import os

key = os.environ.get('polo_key')
sec = os.environ.get('polo_sec')


class App(AsyncApp):
    def ticker(self, **kwargs):
        self.logger.info(kwargs)

    def trades(self, **kwargs):
        self.logger.info(kwargs)

    async def main(self):
        # gets order book updates
        # could be good for real-time stuff, but a bit much for now too
        self.push.subscribe(topic="BTC_ETH", handler=self.trades)
        # this gets every trade there is...honestly it's a bit much
        # because there are so many tickers
        # self.push.subscribe(topic="ticker", handler=self.ticker)
        # volume = await self.public.return24hVolume()

        # self.logger.info(volume)


app = App(api_key=key, api_sec=sec)
app.run()
