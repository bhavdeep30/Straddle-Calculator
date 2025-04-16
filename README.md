# Ultron Options Straddle Strategy Calculator

An interactive options straddle strategy calculator with a futuristic Ultron spaceship theme. This application allows users to analyze the potential profit/loss and breakeven points for options straddle strategies using real market data.

## Features

- Real-time stock data fetching using yfinance
- Black-Scholes model for option pricing
- Interactive profit/loss visualization
- Breakeven point calculations
- Ultron-themed futuristic UI design
- Responsive layout for different screen sizes

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and navigate to http://127.0.0.1:8050/

## Usage

1. Enter a valid ticker symbol (e.g., AAPL, MSFT, GOOGL)
2. Set the strike price for your straddle strategy
3. Specify days to expiration
4. Adjust the risk-free rate if needed
5. Optionally override the implied volatility
6. Click "ANALYZE STRATEGY" to see the results

## What is a Straddle Strategy?

A straddle is an options strategy where the trader purchases both a call option and a put option for the same underlying asset, with the same strike price and expiration date. This strategy is used when a trader believes the underlying asset will experience significant price movement but is uncertain about the direction.

## Disclaimer

This tool is for educational purposes only and should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.
