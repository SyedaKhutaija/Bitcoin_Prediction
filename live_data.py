import requests
import pandas as pd
from datetime import datetime

COINGECKO_MARKET_CHART = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'

def fetch_live_data(days=90, vs_currency='usd'):
    """Fetch recent Bitcoin price data from CoinGecko and return a daily DataFrame.

    days: number of days of history to fetch (int or 'max')
    """
    params = {'vs_currency': vs_currency, 'days': str(days)}
    resp = requests.get(COINGECKO_MARKET_CHART, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # data['prices'] is a list of [timestamp_ms, price]
    prices = data.get('prices', [])
    df = pd.DataFrame(prices, columns=['ts', 'price'])
    df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
    # Use the last price of each day as the close
    df_daily = df.groupby('date').agg({'price': 'last'}).reset_index()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.rename(columns={'price': 'close'})
    df_daily = df_daily.set_index('date').asfreq('D')
    df_daily['close'] = df_daily['close'].ffill()
    return df_daily
