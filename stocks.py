import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import calendar
import requests
import pandas as pd
from prophet import Prophet

# Page config
st.set_page_config(page_title="Stock Market Forecaster", layout="wide")

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


# ---- Utility Functions ----
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance's chart endpoint.

    This avoids the crumb/cookie flow used by yfinance's Ticker.history(),
    which can return HTTP 429 (Too Many Requests) in browser-hosted apps.
    Results are cached briefly so Streamlit reruns do not repeatedly hit Yahoo.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Please enter a stock symbol")

    try:
        start_timestamp = calendar.timegm(
            pd.Timestamp(start_date).to_pydatetime().timetuple()
        )
        # Yahoo's period2 is exclusive, so include the selected end date.
        end_timestamp = calendar.timegm(
            (pd.Timestamp(end_date) + pd.Timedelta(days=1))
            .to_pydatetime()
            .timetuple()
        )

        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(
                    YAHOO_CHART_URL.format(symbol=symbol),
                    params={
                        "period1": start_timestamp,
                        "period2": end_timestamp,
                        "interval": "1d",
                        "events": "history",
                        "includeAdjustedClose": "true",
                    },
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                if response.status_code != 429:
                    response.raise_for_status()
                    break
                last_error = "Yahoo Finance is temporarily rate-limiting requests"
            except requests.RequestException as error:
                last_error = str(error)

            if attempt < 2:
                time.sleep(2 ** attempt)

        if response is None or response.status_code == 429:
            raise RuntimeError(last_error or "Yahoo Finance rate limit reached")

        payload = response.json()
        chart = payload.get("chart", {})
        if chart.get("error"):
            description = chart["error"].get("description", "Unknown Yahoo Finance error")
            raise ValueError(description)

        result = (chart.get("result") or [None])[0]
        if not result or not result.get("timestamp"):
            raise ValueError(f"No data found for symbol {symbol}")

        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            },
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
            .tz_convert(None)
            .normalize(),
        )
        df = df.dropna(subset=["Close"])

        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate key financial metrics from stock data
    """
    try:
        if len(df) < 1:
            raise ValueError("Insufficient data to calculate metrics")
            
        current_price = df['Close'].iloc[-1]
        volume = df['Volume'].iloc[-1]
        
        # Calculate daily change if we have at least 2 rows
        if len(df) >= 2:
            previous_price = df['Close'].iloc[-2]
            daily_change = ((current_price - previous_price) / previous_price) * 100
        else:
            daily_change = 0.0  # No change available for single data point
        
        # Calculate daily trading volume in dollars (price × volume)
        daily_turnover = current_price * volume / 1e9  # in billions
        
        return {
            'current_price': current_price,
            'daily_change': daily_change,
            'volume': volume,
            'daily_turnover': daily_turnover
        }
    except Exception as e:
        raise Exception(f"Failed to calculate metrics: {str(e)}")

@st.cache_data(show_spinner=False)
def create_forecast(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    """
    Create price forecast using Prophet
    """
    try:
        # Prepare data for Prophet
        df_copy = df.copy()
        
        # Convert index to naive datetime (remove timezone info)
        try:
            df_copy.index = pd.to_datetime(df_copy.index).tz_localize(None)
        except:
            # If already timezone-naive or conversion fails, just ensure it's datetime
            df_copy.index = pd.to_datetime(df_copy.index)

        # Reset index and prepare for Prophet
        df_reset = df_copy.reset_index()
        
        # The first column after reset_index should be the date
        # and we know we have 'Close' column
        prophet_df = pd.DataFrame({
            'ds': df_reset.iloc[:, 0],  # First column (date)
            'y': df_reset['Close']      # Close price
        })

        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality='auto',
            yearly_seasonality='auto', 
            weekly_seasonality='auto',
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        # Make future dataframe and predict
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return forecast
    except Exception as e:
        raise Exception(f"Failed to create forecast: {str(e)}")

# ---- Main Application ----
# Title
st.title("Stock Market Forecasting")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")

    # Stock symbol input
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL").upper()

    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_range = st.date_input("Select Date Range",
                            value=(start_date.date(), end_date.date()),
                            max_value=end_date.date())

    # Forecast period
    forecast_days = st.slider(
        "Forecast Days",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        key="forecast_days",
        help="Choose how many calendar days to show in the forecast.",
    )

# Main content
try:
    # Handle date range properly
    if len(date_range) == 2:
        start_dt, end_dt = date_range[0], date_range[1]
    else:
        # If only one date is selected, use it as start and today as end
        start_dt = date_range[0] if len(date_range) == 1 else start_date.date()
        end_dt = end_date.date()
    
    # Fetch stock data
    df = get_stock_data(symbol, start_dt, end_dt)

    # Display current price and metrics
    metrics = calculate_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${metrics['current_price']:.2f}")
    with col2:
        st.metric("Daily Change", f"{metrics['daily_change']:.2f}%")
    with col3:
        st.metric("Volume", f"{metrics['volume']:,.0f}")
    with col4:
        st.metric("Daily Turnover", f"${metrics['daily_turnover']:,.1f}B")

    # Stock price chart
    st.subheader("Stock Price History")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Candlestick"))
    fig.update_layout(xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Volume chart
    st.subheader("Trading Volume")
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume"))
    volume_fig.update_layout(xaxis_title="Date",
                          yaxis_title="Volume",
                          height=300)
    st.plotly_chart(volume_fig, use_container_width=True)

    # Forecast
    st.subheader(f"Price Forecast ({forecast_days} days)")
    # Fit the model once for the maximum supported horizon. The selected
    # horizon is then sliced below, making the slider update immediately.
    full_forecast_df = create_forecast(df, 90)
    last_historical_date = pd.Timestamp(df.index.max()).tz_localize(None)
    forecast_end_date = last_historical_date + pd.Timedelta(days=forecast_days)
    forecast_df = full_forecast_df[
        (full_forecast_df["ds"] > last_historical_date)
        & (full_forecast_df["ds"] <= forecast_end_date)
    ].copy()

    if forecast_df.empty:
        raise ValueError("Unable to generate future forecast values")

    forecast_fig = go.Figure()
    # Historical data
    forecast_fig.add_trace(
        go.Scatter(x=df.index,
                y=df['Close'],
                name="Historical",
                line=dict(color='blue')))
    # Selected future forecast only
    forecast_fig.add_trace(
        go.Scatter(x=forecast_df["ds"],
                y=forecast_df["yhat"],
                name="Forecast",
                line=dict(color='red', dash='dash')))
    # Confidence interval
    forecast_fig.add_trace(
        go.Scatter(x=forecast_df["ds"],
                y=forecast_df["yhat_upper"],
                fill=None,
                mode='lines',
                line_color='rgba(255,0,0,0)',
                showlegend=False))
    forecast_fig.add_trace(
        go.Scatter(x=forecast_df["ds"],
                y=forecast_df["yhat_lower"],
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,0,0,0)',
                name="Confidence Interval"))
    forecast_fig.update_layout(xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=400)
    # Use a shape instead of Plotly's add_vline helper. The helper performs
    # unsupported integer arithmetic with pandas Timestamps in some Plotly
    # versions.
    forecast_start_datetime = last_historical_date.to_pydatetime()
    forecast_fig.add_shape(
        type="line",
        x0=forecast_start_datetime,
        x1=forecast_start_datetime,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", dash="dot"),
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check the stock symbol and try again.")
