import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from pmdarima import auto_arima
from live_data import fetch_live_data
import io
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.set_page_config(page_title='Bitcoin Live Forecast', layout='wide')


@st.cache_data(ttl=30)
def get_data(days):
    return fetch_live_data(days=days)


def sma(series, window):
    return series.rolling(window=window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_indicators(df, sma_windows, ema_spans):
    ind = pd.DataFrame(index=df.index)
    ind['close'] = df['close']
    for w in sma_windows:
        ind[f'SMA_{w}'] = sma(ind['close'], w)
    for s in ema_spans:
        ind[f'EMA_{s}'] = ema(ind['close'], s)
    ind['Volatility'] = ind['close'].pct_change().rolling(window=14).std() * np.sqrt(365)
    return ind


def forecast_prophet(df, horizon):
    df_prophet = df.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    forecast = forecast.set_index('ds')
    return forecast[['yhat', 'yhat_lower', 'yhat_upper']]


def forecast_arima(series, horizon):
    model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    pred = model.predict(n_periods=horizon)
    last_date = series.index[-1]
    idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    return pd.DataFrame({'yhat': pred}, index=idx)


def to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()


st.title('Bitcoin Live Forecast Dashboard')

with st.sidebar:
    st.header('Controls')
    days = st.slider('History (days)', min_value=60, max_value=365*3, value=365)
    horizon = st.slider('Forecast horizon (days)', min_value=1, max_value=90, value=14)
    models = st.multiselect('Models', options=['Prophet', 'ARIMA', 'Ensemble'], default=['Prophet', 'Ensemble'])
    sma_windows = st.multiselect('SMA windows', options=[7, 14, 30, 90], default=[7, 30])
    ema_spans = st.multiselect('EMA spans', options=[7, 14, 30], default=[14])
    refresh_interval = st.slider('Auto-refresh (seconds)', min_value=0, max_value=300, value=60)
    price_alert = st.number_input('Price alert if above (USD)', min_value=0.0, value=0.0)
    if st.button('Clear cache'):
        get_data.clear()

col1, col2 = st.columns([3, 1])

with st.spinner('Fetching data...'):
    df = get_data(days)

latest_price = df['close'].iloc[-1]
with col2:
    st.metric('Latest BTC Price (USD)', f"${latest_price:,.2f}")
    st.write('Volatility (annualized recent):', f"{df['close'].pct_change().rolling(14).std().iloc[-1]*np.sqrt(365):.4f}")
    if price_alert and latest_price > price_alert:
        st.success(f'Price crossed above alert: ${price_alert:,.2f}')

ind = compute_indicators(df, sma_windows, ema_spans)

forecast_frames = []
if 'Prophet' in models:
    with st.spinner('Training Prophet...'):
        f_p = forecast_prophet(df, horizon)
        forecast_frames.append(f_p.rename(columns={'yhat': 'Prophet'}))

if 'ARIMA' in models:
    with st.spinner('Training ARIMA...'):
        f_a = forecast_arima(df['close'], horizon)
        forecast_frames.append(f_a.rename(columns={'yhat': 'ARIMA'}))

ensemble = None
if 'Ensemble' in models and forecast_frames:
    # align and average available forecasts
    combined = pd.concat(forecast_frames, axis=1)
    combined['Ensemble'] = combined.mean(axis=1)
    ensemble = combined[['Ensemble']]

# Combine forecasts for plotting
plot_forecast = pd.DataFrame()
for fr in forecast_frames:
    plot_forecast = pd.concat([plot_forecast, fr], axis=1)
if ensemble is not None:
    plot_forecast = pd.concat([plot_forecast, ensemble], axis=1)

# Plot interactive chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Historical', line=dict(color='black')))
for col in plot_forecast.columns:
    fig.add_trace(go.Scatter(x=plot_forecast.index, y=plot_forecast[col], mode='lines', name=col))
for w in sma_windows:
    fig.add_trace(go.Scatter(x=ind.index, y=ind[f'SMA_{w}'], mode='lines', name=f'SMA {w}', line=dict(dash='dot')))
for s in ema_spans:
    fig.add_trace(go.Scatter(x=ind.index, y=ind[f'EMA_{s}'], mode='lines', name=f'EMA {s}', line=dict(dash='dash')))

fig.update_layout(title='Bitcoin Price with Forecasts and Indicators', xaxis_title='Date', yaxis_title='Price (USD)', height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader('Forecast Table (tail)')
if not plot_forecast.empty:
    st.dataframe(plot_forecast.tail(horizon))

st.subheader('Download')
csv_bytes = to_csv_bytes(pd.concat([df['close'], ind.drop(columns=['close']), plot_forecast], axis=1))
st.download_button('Download CSV', data=csv_bytes, file_name='btc_forecast.csv', mime='text/csv')

st.markdown('---')
st.markdown('Notes: Data from CoinGecko. ARIMA may take time to fit for long histories. Ensemble averages available forecasts.')
