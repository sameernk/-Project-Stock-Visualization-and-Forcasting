# 📈 Stock Market Forecaster

An interactive web application built with **Streamlit** that fetches real-time and historical stock market data, visualizes price trends, and generates future price forecasts using Facebook's **Prophet** time-series forecasting model.

## Features

- **Live Stock Data** — Fetches historical OHLCV (Open, High, Low, Close, Volume) data directly from Yahoo Finance's chart API for any valid stock symbol.
- **Key Metrics Dashboard** — Displays current price, daily percentage change, trading volume, and daily turnover (in billions USD) at a glance.
- **Interactive Candlestick Chart** — Visualizes historical price action using Plotly's candlestick charts.
- **Volume Chart** — Bar chart showing trading volume over the selected date range.
- **Price Forecasting** — Uses Prophet to model and predict future stock prices (7–90 days) with confidence intervals, plotted alongside historical data.
- **Configurable Date Range & Forecast Horizon** — Users can select a custom historical date range and adjust the forecast period via a slider.
- **Resilient Data Fetching** — Includes retry logic with exponential backoff to gracefully handle Yahoo Finance rate limits (HTTP 429).
- **Caching** — Uses Streamlit's `@st.cache_data` to avoid redundant API calls and speed up reruns.

## Tech Stack

| Component        | Technology |
|-------------------|------------|
| Frontend/App      | Streamlit |
| Charts            | Plotly (Graph Objects) |
| Forecasting       | Prophet |
| Data Source       | Yahoo Finance (Chart API) |
| Data Handling     | Pandas |
| HTTP Requests     | Requests |

## How It Works

1. **Data Fetching** (`get_stock_data`) — Queries Yahoo Finance's `/v8/finance/chart/{symbol}` endpoint directly (bypassing `yfinance`'s cookie/crumb flow) with retry-with-backoff logic to handle rate limiting.
2. **Metrics Calculation** (`calculate_metrics`) — Computes current price, daily % change, volume, and daily turnover from the fetched data.
3. **Forecasting** (`create_forecast`) — Prepares the data for Prophet, fits a model with automatic seasonality detection, and generates a 90-day forecast (sliced based on user-selected horizon for instant UI updates).
4. **Visualization** — Renders candlestick, volume, and forecast charts (with confidence intervals) using Plotly.

## Installation

```bash
pip install streamlit plotly pandas requests prophet
```

## Usage

```bash
streamlit run app.py
```

Then, in the sidebar:
1. Enter a stock ticker symbol (e.g., `AAPL`, `TSLA`, `GOOGL`)
2. Select a historical date range
3. Adjust the forecast horizon (7–90 days) using the slider

## Project Structure

```
├── app.py          # Main Streamlit application
└── README.md
```

## Notes

- Forecasts are generated using historical closing prices only and are intended for **educational/demonstrative purposes**, not financial advice.
- Yahoo Finance's public endpoint may occasionally rate-limit requests; the app automatically retries with exponential backoff before surfacing an error.

## License

This project is open source and available for personal and educational use.
