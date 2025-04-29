import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import ta
import numpy as np

# Fetch stock data based on ticker, period, & interval through Yahoo Finance API
def fetch_stock_data(ticker, period, interval):
    try:
        # Tarih aralığını hesapla
        end_date = datetime.now()
        if period == '1wk':
            start_date = end_date - timedelta(days=7)
        elif period.isdigit():
            start_date = end_date - timedelta(days=int(period[:-1]))
        else:
            start_date = None  # 'max' gibi durumlar için None bırak

        # Veri çekme
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
            return None
        return data
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data: {e}")
        return None

# Format the date & time to ensure it is timezone aware with correct formatting
def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Calculate basic metrics from stock data
def calculate_metrics(data):
    last_close = float(data['Close'].iloc[-1])  # Ensure it's a float
    prev_close = float(data['Close'].iloc[0])   # Ensure it's a float
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = float(data['High'].max())            # Ensure it's a float
    low = float(data['Low'].min())              # Ensure it's a float
    volume = int(data['Volume'].sum())          # Ensure it's an int
    return last_close, change, pct_change, high, low, volume

# Add technical indicators (SMA, EMA, RSI)
def add_technical_indicators(data):
    # Ensure 'Close' is a Series, not a DataFrame
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()  # Convert to Series if it's a DataFrame

    # Add technical indicators
    data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
    data['RSI_14'] = ta.momentum.rsi(close_series, window=14)
    return data

# Dashboard app page layout
st.set_page_config(layout='wide')
st.title('Real-Time Stock Dashboard')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'AAPL').strip().upper()
if not ticker:
    st.sidebar.error("Please enter a valid ticker symbol.")
time_period = st.sidebar.selectbox('Time Period', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20', 'RSI 14'])

# Interval Mapping
interval_mapping = {
    '1d': '1m',
    '5d': '5m',
    '1mo': '1h',
    '3mo': '1d',
    '6mo': '1d',
    '1y': '1wk',
    '5y': '1mo',
    'max': '1mo',
}

# Update dashboard based on user inputs
if st.sidebar.button('Update'):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data is not None:
        data = process_data(data)
        data = add_technical_indicators(data)

        last_close, change, pct_change, high, low, volume = calculate_metrics(data)

        # Display metrics
        st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")
        col1, col2, col3 = st.columns(3)
        col1.metric('High', f"{high:.2f} USD")
        col2.metric('Low', f"{low:.2f} USD")
        col3.metric('Volume', f"{volume:,}")

        # Plot the Stock Price Chart
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        # Add selected technical indicators to chart
        for indicator in indicators:
            if indicator == 'SMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
            elif indicator == 'EMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))
            elif indicator == 'RSI 14':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI_14'], name='RSI 14', yaxis="y2"))

        # Formatting of the chart
        fig.update_layout(title=f"{ticker} {time_period.upper()} Chart",
                          xaxis_title='Time',
                          yaxis_title='Price (USD)',
                          yaxis2=dict(title='RSI', overlaying='y', side='right', showgrid=False),
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Display historical data & technical indicators
        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        st.subheader('Technical Indicators')
        st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20', 'RSI_14']])

# Real-time stock prices of selected symbols in sidebar
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if real_time_data is not None:
        real_time_data = process_data(real_time_data)
        last_price = float(real_time_data['Close'].iloc[-1])
        change = float(last_price - real_time_data['Open'].iloc[0])
        pct_change = float((change / real_time_data['Open'].iloc[0]) * 100)
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides real-time stock data and technical indicators for various time periods.')
