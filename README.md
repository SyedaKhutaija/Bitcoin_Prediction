# Bitcoin Live Forecast

This workspace contains scripts to run Bitcoin forecasting locally and as a live dashboard.

Quick start:

1. Install dependencies:

```
python -m pip install -r requirements.txt
```

2. Run the Streamlit dashboard (live data):

```
streamlit run app_streamlit.py
```

3. Run the historical forecasting script (optionally using live data):

```
python bitcoin_forecast.py            # uses local cleaned_bitcoin_data.xlsx
python bitcoin_forecast.py --live    # fetches live data from CoinGecko
```

Notes:
- The Streamlit app caches live API responses for 60 seconds. Use the Refresh button to force retrain.
- The `live_data.py` module fetches daily close prices from CoinGecko.
