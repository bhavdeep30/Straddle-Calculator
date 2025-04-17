import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime
import json

# Initialize the Dash app with a dark theme
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Ultron color scheme
colors = {
    'background': '#0a0e17',
    'text': '#e0e0e0',
    'accent': '#ff0000',  # Ultron red
    'secondary': '#3a506b',
    'profit': '#00ff7f',
    'loss': '#ff4757',
    'grid': '#1e2a3a',
    'panel': '#141c26'
}

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free interest rate (as decimal, e.g., 0.05 for 5%)
    sigma: Volatility (as decimal, e.g., 0.3 for 30%)
    option_type: "call" or "put"
    
    Returns:
    Option price
    """
    # Handle edge cases
    if T <= 0:
        # At expiration
        if option_type == "call":
            return max(0, S - K)
        else:  # put
            return max(0, K - S)
    
    if sigma <= 0:
        sigma = 0.001  # Avoid division by zero
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(0, price)  # Option price can't be negative

# Calculate option Greeks
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option Greeks using Black-Scholes model
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free interest rate (as decimal)
    sigma: Volatility (as decimal)
    option_type: "call" or "put"
    
    Returns:
    Dictionary with Delta, Gamma, Theta, Vega, and Rho
    """
    if T <= 0:
        return {
            "delta": 1.0 if option_type == "call" and S > K else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }
    
    if sigma <= 0:
        sigma = 0.001  # Avoid division by zero
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate common terms
    sqrt_t = np.sqrt(T)
    nd1 = norm.pdf(d1)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = nd1 / (S * sigma * sqrt_t)
    
    # Theta
    term1 = -S * nd1 * sigma / (2 * sqrt_t)
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = term1 + term2
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = term1 + term2
    
    # Vega (same for calls and puts)
    vega = S * sqrt_t * nd1 / 100  # Divided by 100 to get change per 1% vol change
    
    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,  # Daily theta
        "vega": vega,
        "rho": rho
    }

# Calculate theoretical future option prices for different stock prices
def calculate_theoretical_prices(current_price, straddle_strike, straddle_put_strike, 
                                strangle_call_strike, strangle_put_strike, days_to_expiry, 
                                risk_free_rate, implied_volatility_call, implied_volatility_put, 
                                price_range):
    """
    Calculate theoretical option prices for different stock prices
    
    Parameters:
    current_price: Current stock price
    straddle_strike: Strike price for straddle (both call and put)
    straddle_put_strike: Strike price for straddle put (should be same as straddle_strike)
    strangle_call_strike: Strike price for strangle call
    strangle_put_strike: Strike price for strangle put
    days_to_expiry: Days to expiration
    risk_free_rate: Risk-free interest rate (as percentage)
    implied_volatility_call: IV for call option (as percentage)
    implied_volatility_put: IV for put option (as percentage)
    price_range: Array of stock prices to calculate theoretical prices
    
    Returns:
    DataFrame with stock prices and corresponding theoretical option prices
    """
    # Convert to decimal
    r = risk_free_rate / 100
    iv_call = implied_volatility_call / 100
    iv_put = implied_volatility_put / 100
    
    # Convert days to years
    T = max(0.0001, days_to_expiry / 365)
    
    theoretical_calls = []
    theoretical_puts = []
    
    # Determine if it's a straddle or strangle
    is_straddle = straddle_strike is not None
    
    if is_straddle:
        call_strike = straddle_strike
        put_strike = straddle_strike
    else:
        call_strike = strangle_call_strike
        put_strike = strangle_put_strike
    
    for price in price_range:
        call_price = black_scholes(price, call_strike, T, r, iv_call, "call")
        put_price = black_scholes(price, put_strike, T, r, iv_put, "put")
        
        theoretical_calls.append(call_price)
        theoretical_puts.append(put_price)
    
    return pd.DataFrame({
        'Stock Price': price_range,
        'Call Price': theoretical_calls,
        'Put Price': theoretical_puts
    })

# Calculate straddle strategy profit at different stock prices
def calculate_straddle_profit(current_price, call_strike, put_strike, call_price, put_price, price_range, is_straddle):
    """
    Calculate profit/loss for a straddle/strangle strategy at different stock prices
    
    Parameters:
    current_price: Current stock price
    call_strike: Strike price for call option
    put_strike: Strike price for put option
    call_price: Premium paid for call option
    put_price: Premium paid for put option
    price_range: Array of stock prices to calculate profit/loss
    is_straddle: Boolean indicating if this is a true straddle
    
    Returns:
    DataFrame with stock prices and corresponding profit/loss
    """
    total_premium = call_price + put_price
    
    profits = []
    call_profits = []
    put_profits = []
    contract_profits = []
    is_breakeven = []  # Track if price is a breakeven point
    
    # Calculate breakeven points
    lower_breakeven, upper_breakeven = calculate_breakeven_points(
        call_strike, put_strike, call_price, put_price, is_straddle
    )
    
    for price in price_range:
        # Calculate option profits
        call_profit = max(0, price - call_strike) - call_price
        put_profit = max(0, put_strike - price) - put_price
        total_profit = call_profit + put_profit
        
        # Calculate contract value (multiply by 100)
        contract_profit = total_profit * 100
        
        # Check if this price is a breakeven point (within $0.01)
        is_be = abs(total_profit) < 0.01
        
        profits.append(total_profit)
        call_profits.append(call_profit)
        put_profits.append(put_profit)
        contract_profits.append(contract_profit)
        is_breakeven.append(is_be)
    
    # Create DataFrame with additional breakeven info
    profit_df = pd.DataFrame({
        'Stock Price': price_range,
        'Profit/Loss': profits,
        'Call P/L': call_profits,
        'Put P/L': put_profits,
        'Contract P/L': contract_profits,
        'Is Breakeven': is_breakeven
    })
    
    # Add breakeven points explicitly if they're not in the price range
    be_points = [lower_breakeven, upper_breakeven]
    for be_price in be_points:
        if be_price not in profit_df['Stock Price'].values:
            call_profit = max(0, be_price - call_strike) - call_price
            put_profit = max(0, put_strike - be_price) - put_price
            total_profit = call_profit + put_profit
            
            profit_df.loc[len(profit_df)] = {
                'Stock Price': be_price,
                'Profit/Loss': total_profit,
                'Call P/L': call_profit,
                'Put P/L': put_profit,
                'Contract P/L': total_profit * 100,
                'Is Breakeven': True
            }
    
    # Sort by stock price and reset index
    profit_df = profit_df.sort_values('Stock Price').reset_index(drop=True)
    
    return profit_df

# Calculate breakeven points for straddle/strangle
def calculate_breakeven_points(call_strike, put_strike, call_price, put_price, is_straddle):
    """
    Calculate exact breakeven points where Contract P/L is zero
    
    Parameters:
    call_strike: Strike price for call option
    put_strike: Strike price for put option
    call_price: Premium paid for call option
    put_price: Premium paid for put option
    is_straddle: Boolean indicating if this is a true straddle
    
    Returns:
    Tuple of lower and upper breakeven points
    """
    if is_straddle:
        # For straddle (same strike price)
        strike = call_strike  # same as put_strike
        total_premium = call_price + put_price
        lower_breakeven = strike - total_premium
        upper_breakeven = strike + total_premium
    else:
        # For strangle (different strikes)
        lower_breakeven = put_strike - (call_price + put_price)
        upper_breakeven = call_strike + (call_price + put_price)
    
    return lower_breakeven, upper_breakeven

# Function to get options chain for a ticker
def get_options_chain(ticker, selected_expiration=None):
    """
    Get options chain for a ticker using minimal yfinance calls
    
    Parameters:
    ticker: Stock ticker symbol
    selected_expiration: Optional specific expiration date to fetch
    
    Returns:
    calls_df, puts_df, exp_date, all_expirations, current_price
    """
    try:
        # Normalize ticker
        ticker = ticker.strip().upper()
        
        # Create ticker object with force_refresh to avoid cached data
        ticker_obj = yf.Ticker(ticker)
        
        # Get current price first to ensure connection works
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not retrieve price data for {ticker}")
        current_price = hist['Close'].iloc[-1]
        
        # Get available expiration dates
        expirations = ticker_obj.options
        
        if not expirations or len(expirations) == 0:
            raise ValueError(f"No options data available for {ticker}")
        
        # Use selected expiration or default to first available
        exp_date = selected_expiration if selected_expiration in expirations else expirations[0]
        
        # Get options for this expiration with a retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                options = ticker_obj.option_chain(exp_date)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to fetch option chain after {max_retries} attempts: {e}")
                print(f"Retry {attempt+1}/{max_retries} fetching options for {ticker}")
                import time
                time.sleep(1)  # Wait before retrying
        
        # Format the data
        calls = options.calls
        puts = options.puts
        
        if calls.empty or puts.empty:
            raise ValueError(f"No options chain data available for {ticker} on {exp_date}")
        
        # Format the data - handle potential missing columns gracefully
        required_columns = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'inTheMoney']
        
        # Check if all required columns exist
        for col in required_columns:
            if col not in calls.columns or col not in puts.columns:
                print(f"Warning: Missing column {col} in options data")
                # Create the column with default values if missing
                if col not in calls.columns:
                    calls[col] = 0.0
                if col not in puts.columns:
                    puts[col] = 0.0
        
        # Now safely select the columns
        calls = calls[required_columns]
        calls.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'IV', 'ITM']
        calls['IV'] = (calls['IV'] * 100).round(2)
        
        puts = puts[required_columns]
        puts.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'IV', 'ITM']
        puts['IV'] = (puts['IV'] * 100).round(2)
        
        # Get expiration index
        current_exp_index = expirations.index(exp_date)
        prev_exp = expirations[current_exp_index - 1] if current_exp_index > 0 else None
        next_exp = expirations[current_exp_index + 1] if current_exp_index < len(expirations) - 1 else None
        
        # Format expiration dates for display
        exp_date_formatted = datetime.datetime.strptime(exp_date, "%Y-%m-%d").strftime("%B %d, %Y")
        
        return calls, puts, exp_date, expirations, current_price, exp_date_formatted, prev_exp, next_exp
    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
        # Return more detailed error information
        error_msg = str(e)
        return None, None, None, None, None, f"Error: {error_msg}", None, None

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    # Store for Black-Scholes calculations
    dcc.Store(id='bs-calculations', storage_type='memory'),
    # Header
    html.Div(style={'padding': '20px', 'textAlign': 'center', 'borderBottom': f'1px solid {colors["secondary"]}'}, children=[
        html.H1("ULTRON STRADDLE STRATEGY ANALYZER", style={'color': colors['accent'], 'fontWeight': 'bold', 'letterSpacing': '2px'}),
        html.H3("Advanced Options Trading Intelligence System", style={'color': colors['text'], 'fontStyle': 'italic'})
    ]),
    
    # Store components for selected options
    dcc.Store(id='selected-call', storage_type='memory'),
    dcc.Store(id='selected-put', storage_type='memory'),
    dcc.Store(id='stock-price-store', storage_type='memory'),
    dcc.Store(id='current-ticker', storage_type='memory'),
    dcc.Store(id='price-scenarios', storage_type='memory'),
    
    # Loading component
    dcc.Loading(
        id="loading-data",
        type="circle",
        color=colors['accent'],
        children=[html.Div(id="loading-output")],
        style={'position': 'fixed', 'top': '50%', 'left': '50%', 'transform': 'translate(-50%, -50%)', 'zIndex': '1000'}
    ),
    
    # Main content
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
        # Left panel - Ticker input and stock info
        html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, children=[
            html.H3("SELECT TARGET", style={'color': colors['accent'], 'borderBottom': f'1px solid {colors["secondary"]}', 'paddingBottom': '10px'}),
            
            # Ticker input
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("TICKER SYMBOL", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                html.Div(style={'display': 'flex'}, children=[
                    dcc.Input(
                        id='ticker-input',
                        type='text',
                        value='AAPL',
                        style={'flex': '1', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px 0 0 5px'}
                    ),
                    html.Button(
                        'FETCH OPTIONS',
                        id='fetch-options-button',
                        style={
                            'backgroundColor': colors['secondary'],
                            'color': colors['text'],
                            'border': 'none',
                            'padding': '10px 15px',
                            'borderRadius': '0 5px 5px 0',
                            'cursor': 'pointer',
                            'fontWeight': 'bold'
                        }
                    ),
                ]),
            ]),
            
            # Risk-free rate
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("RISK-FREE RATE (%)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(
                    id='risk-free-rate',
                    type='number',
                    value=4.5,
                    min=0,
                    max=20,
                    step=0.1,
                    style={'width': '100%', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px'}
                ),
            ]),
            
            # Stock info display
            html.Div(id='stock-info', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '5px', 'border': f'1px solid {colors["secondary"]}'}),
            
            # Selected options display
            html.Div(id='selected-options-display', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '5px', 'border': f'1px solid {colors["secondary"]}'}),
            
            # Calculate button
            html.Button(
                'ANALYZE STRATEGY',
                id='calculate-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': 'none',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'marginTop': '20px'
                }
            ),
        ]),
        
        # Middle panel - Options tables
        html.Div(style={'flex': '2', 'minWidth': '500px', 'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, children=[
            html.H3("OPTIONS CHAIN", style={'color': colors['accent'], 'borderBottom': f'1px solid {colors["secondary"]}', 'paddingBottom': '10px'}),
            
            # Expiration date navigation
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '15px'}, children=[
                html.Button(
                    '◀ PREV',
                    id='prev-expiry-button',
                    style={
                        'backgroundColor': colors['secondary'],
                        'color': colors['text'],
                        'border': 'none',
                        'padding': '8px 15px',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'width': '100px'
                    }
                ),
                html.Div(id='expiration-display', style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Button(
                    'NEXT ▶',
                    id='next-expiry-button',
                    style={
                        'backgroundColor': colors['secondary'],
                        'color': colors['text'],
                        'border': 'none',
                        'padding': '8px 15px',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'width': '100px'
                    }
                ),
            ]),
            
            # Store components for expiration navigation
            dcc.Store(id='prev-expiry', storage_type='memory'),
            dcc.Store(id='current-expiry', storage_type='memory'),
            dcc.Store(id='next-expiry', storage_type='memory'),
            dcc.Store(id='all-expiries', storage_type='memory'),
            
            # Options tables
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
                # Calls table
                html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4("CALLS", style={'color': colors['profit'], 'textAlign': 'center'}),
                    html.Div(id='calls-table-container', style={'overflowX': 'auto'})
                ]),
                
                # Puts table
                html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                    html.H4("PUTS", style={'color': colors['loss'], 'textAlign': 'center'}),
                    html.Div(id='puts-table-container', style={'overflowX': 'auto'})
                ]),
            ]),
        ]),
        
        # Right panel - Results and Graph
        html.Div(style={'flex': '2', 'minWidth': '500px', 'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, children=[
            html.H3("STRATEGY ANALYSIS", style={'color': colors['accent'], 'borderBottom': f'1px solid {colors["secondary"]}', 'paddingBottom': '10px'}),
            
            # Strategy results
            html.Div(id='strategy-results', style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '5px', 'border': f'1px solid {colors["secondary"]}'}),
            
            # Black-Scholes Pricing Scenarios
            html.Div([
                html.H4("BLACK-SCHOLES PRICING SCENARIOS", style={'color': colors['accent'], 'marginTop': '20px', 'marginBottom': '10px'}),
                
                # Black-Scholes date display
                html.Div(style={'marginBottom': '15px', 'textAlign': 'center'}, children=[
                    html.Div(id='bs-date-display', style={
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'fontSize': '16px',
                        'marginBottom': '15px'
                    }),
                ]),
                
                # Black-Scholes pricing table
                html.Div(id='bs-pricing-table', style={'marginBottom': '20px', 'overflowX': 'auto'}),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '5px', 'border': f'1px solid {colors["secondary"]}'}),
            
            # Profit/Loss graph with improved config
            dcc.Graph(
                id='profit-loss-graph',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'straddle_analysis',
                        'height': 800,
                        'width': 1200,
                        'scale': 2
                    }
                },
                style={'backgroundColor': colors['background'], 'height': '600px'}
            )
        ]),
    ]),
    
    # Footer
    html.Div(style={'padding': '20px', 'textAlign': 'center', 'borderTop': f'1px solid {colors["secondary"]}'}, children=[
        html.P("ULTRON FINANCIAL SYSTEMS © 2025", style={'color': colors['secondary']}),
        html.P("DISCLAIMER: This tool is for educational purposes only. Not financial advice.", style={'color': colors['secondary'], 'fontSize': '12px'})
    ])
])


# Callback to fetch options and display tables
@app.callback(
    [Output('calls-table-container', 'children'),
     Output('puts-table-container', 'children'),
     Output('stock-info', 'children'),
     Output('expiration-display', 'children'),
     Output('stock-price-store', 'data'),
     Output('loading-output', 'children'),
     Output('current-ticker', 'data'),
     Output('prev-expiry', 'data'),
     Output('current-expiry', 'data'),
     Output('next-expiry', 'data'),
     Output('all-expiries', 'data')],
    [Input('fetch-options-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_options_tables(n_clicks, ticker):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not ticker or ticker.strip() == "":
        return (
            html.Div("Please enter a valid ticker symbol", style={'color': colors['loss']}),
            html.Div("Please enter a valid ticker symbol", style={'color': colors['loss']}),
            html.Div("No stock data available", style={'color': colors['loss']}),
            "No expiration dates available",
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
    
    try:
        # Show loading message
        loading_message = html.Div("Fetching options data...", style={'color': colors['text']})
        
        # Normalize ticker
        ticker = ticker.strip().upper()
        
        # Get options chain with improved approach
        calls_df, puts_df, exp_date, all_expirations, current_price, exp_date_formatted, prev_exp, next_exp = get_options_chain(ticker)
        
        if calls_df is None or puts_df is None:
            error_message = exp_date_formatted if isinstance(exp_date_formatted, str) and exp_date_formatted.startswith("Error") else "Could not retrieve options data"
            return (
                html.Div("No options data available", style={'color': colors['loss']}),
                html.Div("No options data available", style={'color': colors['loss']}),
                html.Div([
                    html.H4(f"{ticker}", style={'color': colors['accent'], 'marginTop': '0'}),
                    html.P(error_message, style={'margin': '5px 0', 'color': colors['loss']})
                ]),
                "No expiration dates available",
                None,
                None,
                ticker,
                None,
                None,
                None,
                None
            )
        
        # Find the closest strikes to current price
        closest_call_idx = (calls_df['Strike'] - current_price).abs().idxmin()
        closest_put_idx = (puts_df['Strike'] - current_price).abs().idxmin()
        
        # Sort the dataframes by strike price
        calls_df = calls_df.sort_values('Strike')
        puts_df = puts_df.sort_values('Strike')
        
        # Get 5 strikes above and below the closest strike
        call_start_idx = max(0, closest_call_idx - 5)
        call_end_idx = min(len(calls_df) - 1, closest_call_idx + 5)
        put_start_idx = max(0, closest_put_idx - 5)
        put_end_idx = min(len(puts_df) - 1, closest_put_idx + 5)
        
        # Filter the dataframes to show options around current price
        visible_calls_df = calls_df.iloc[call_start_idx:call_end_idx + 1].copy()
        visible_puts_df = puts_df.iloc[put_start_idx:put_end_idx + 1].copy()
        
        # Add a column to highlight the row closest to current price
        visible_calls_df['Near Current'] = False
        visible_calls_df.loc[closest_call_idx, 'Near Current'] = True
        
        visible_puts_df['Near Current'] = False
        visible_puts_df.loc[closest_put_idx, 'Near Current'] = True
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in visible_calls_df.columns if col != 'Near Current'
            ],
            data=visible_calls_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(0, 255, 127, 0.2)',
                    'color': colors['profit']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in visible_puts_df.columns if col != 'Near Current'
            ],
            data=visible_puts_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(255, 71, 87, 0.2)',
                    'color': colors['loss']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        # Stock info display with current price highlighted
        stock_info = html.Div([
            html.H4(f"{ticker}", style={'color': colors['accent'], 'marginTop': '0'}),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                html.Div([
                    html.P("CURRENT PRICE:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"${current_price:.2f}", style={
                        'margin': '5px 0', 
                        'fontSize': '18px',
                        'backgroundColor': colors['secondary'],
                        'padding': '5px 10px',
                        'borderRadius': '5px',
                        'display': 'inline-block'
                    })
                ])
            ])
        ])
        
        return (
            calls_table, 
            puts_table, 
            stock_info, 
            exp_date_formatted,
            json.dumps({'price': current_price}), 
            None, 
            ticker,
            prev_exp,
            exp_date,
            next_exp,
            json.dumps(all_expirations)
        )
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            html.Div(error_message, style={'color': colors['loss']}),
            html.Div(error_message, style={'color': colors['loss']}),
            html.Div(error_message, style={'color': colors['loss']}),
            [],
            None,
            None,
            None,
            None
        )

# Callback to store selected call option
@app.callback(
    Output('selected-call', 'data'),
    [Input('calls-table', 'selected_rows')],
    [State('calls-table', 'data')]
)
def store_selected_call(selected_rows, data):
    if not selected_rows or not data:
        return None
    
    selected_row = selected_rows[0]
    return data[selected_row]

# Callback to store selected put option
@app.callback(
    Output('selected-put', 'data'),
    [Input('puts-table', 'selected_rows')],
    [State('puts-table', 'data')]
)
def store_selected_put(selected_rows, data):
    if not selected_rows or not data:
        return None
    
    selected_row = selected_rows[0]
    return data[selected_row]

# Callback to display selected options
@app.callback(
    Output('selected-options-display', 'children'),
    [Input('selected-call', 'data'),
     Input('selected-put', 'data')]
)
def update_selected_options_display(call_data, put_data):
    if not call_data and not put_data:
        return html.P("Select options from the tables to analyze", style={'color': colors['secondary']})
    
    selected_options = []
    
    if call_data:
        selected_options.append(
            html.Div([
                html.H5("SELECTED CALL", style={'color': colors['profit'], 'marginTop': '0'}),
                html.P(f"Strike: ${call_data['Strike']}", style={'margin': '2px 0'}),
                html.P(f"Price: ${call_data['Last Price']}", style={'margin': '2px 0'}),
                html.P(f"IV: {call_data['IV']}%", style={'margin': '2px 0'})
            ])
        )
    
    if put_data:
        selected_options.append(
            html.Div([
                html.H5("SELECTED PUT", style={'color': colors['loss'], 'marginTop': '0', 'marginTop': '10px' if call_data else '0'}),
                html.P(f"Strike: ${put_data['Strike']}", style={'margin': '2px 0'}),
                html.P(f"Price: ${put_data['Last Price']}", style={'margin': '2px 0'}),
                html.P(f"IV: {put_data['IV']}%", style={'margin': '2px 0'})
            ])
        )
    
    return html.Div(selected_options)

# Callback for strategy analysis and Black-Scholes calculations
@app.callback(
    [Output('strategy-results', 'children'),
     Output('profit-loss-graph', 'figure'),
     Output('bs-calculations', 'data'),
     Output('bs-date-display', 'children'),
     Output('bs-pricing-table', 'children'),
     Output('price-scenarios', 'data')],
    [Input('calculate-button', 'n_clicks'),
     Input('current-expiry', 'data')],  # Also trigger on expiry date change
    [State('selected-call', 'data'),
     State('selected-put', 'data'),
     State('stock-price-store', 'data'),
     State('risk-free-rate', 'value')]
)
def update_results(n_clicks, expiry_date, call_data, put_data, stock_price_data, risk_free_rate):
    # Get the triggered input
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Only proceed if calculate button was clicked or expiry date changed with valid selections
    if (triggered_id == 'calculate-button' and n_clicks == 0) or not call_data or not put_data or not stock_price_data:
        # Initial state or missing data
        return (
            html.P("Select both a call and put option, then click ANALYZE STRATEGY", style={'color': colors['secondary']}),
            go.Figure(),
            None,
            "Select options first",
            html.Div("No data available"),
            None
        )
    
    try:
        # Get data from selected options
        call_price = call_data['Last Price']
        put_price = put_data['Last Price']
        call_strike = call_data['Strike']
        put_strike = put_data['Strike']
        call_iv = call_data['IV']
        put_iv = put_data['IV']
        current_price = json.loads(stock_price_data)['price']
        
        # Calculate days to expiration
        if expiry_date:
            expiry_date_obj = datetime.datetime.strptime(expiry_date, "%Y-%m-%d")
            days_to_expiry = (expiry_date_obj - datetime.datetime.now()).days
            days_to_expiry = max(0, days_to_expiry)  # Ensure non-negative
        else:
            days_to_expiry = 30  # Default if no expiry date
        
        # Check if we have a true straddle (same strike price)
        is_true_straddle = call_strike == put_strike
        strategy_type = "STRADDLE" if is_true_straddle else "STRANGLE"
        
        # Use the appropriate strike price for calculations
        if is_true_straddle:
            strike_price = call_strike  # Both are the same
        else:
            # For a strangle, we'll use both strikes separately
            call_strike_price = call_strike
            put_strike_price = put_strike
        # Total premium paid
        total_premium = call_price + put_price
        
        # Calculate exact breakeven points where Contract P/L is zero
        lower_breakeven, upper_breakeven = calculate_breakeven_points(
            call_strike, 
            put_strike, 
            call_price, 
            put_price, 
            is_true_straddle
        )
        
        # Calculate profit/loss at different stock prices
        price_range_min = max(0.1, current_price * 0.5)  # Avoid negative or zero prices
        price_range_max = current_price * 1.5
        price_range = np.linspace(price_range_min, price_range_max, 100)
        
        # Generate price scenarios for the table focused around current price
        # Create a tighter range around current price for the table
        table_range_min = max(0.1, current_price * 0.9)  # 10% below current price
        table_range_max = current_price * 1.1  # 10% above current price
        
        # Create more points near the current price
        lower_range = np.linspace(table_range_min, current_price * 0.98, 3)
        middle_range = np.linspace(current_price * 0.99, current_price * 1.01, 5)  # More points around current price
        upper_range = np.linspace(current_price * 1.02, table_range_max, 3)
        
        # Combine the ranges
        price_scenarios = np.unique(np.concatenate([lower_range, middle_range, upper_range]))
        
        # Calculate Black-Scholes theoretical prices for different dates
        days_list = [max(0, days_to_expiry - 14), max(0, days_to_expiry - 7), days_to_expiry, 
                    min(days_to_expiry + 7, 365), min(days_to_expiry + 14, 365)]
        
        bs_results = {}
        for days in days_list:
            # Calculate time to expiry in years
            t_expiry = max(0.0001, days / 365)
            
            # Calculate theoretical prices for different stock prices
            theoretical_prices = calculate_theoretical_prices(
                current_price, 
                call_strike if is_true_straddle else None,  # For true straddle
                put_strike if is_true_straddle else None,   # For true straddle
                call_strike if not is_true_straddle else None,  # For strangle
                put_strike if not is_true_straddle else None,   # For strangle
                days, 
                risk_free_rate, 
                call_iv, 
                put_iv, 
                price_scenarios
            )
            
            bs_results[days] = theoretical_prices.to_dict('records')
        
        # Format expiry date for display
        formatted_expiry_date = "N/A"
        if expiry_date:
            try:
                expiry_date_obj = datetime.datetime.strptime(expiry_date, "%Y-%m-%d")
                formatted_expiry_date = expiry_date_obj.strftime("%B %d, %Y")
            except:
                formatted_expiry_date = expiry_date
        
        # Store Black-Scholes calculations for date navigation
        bs_calculations = {
            'current_days': days_to_expiry,
            'days_list': days_list,
            'results': bs_results,
            'call_price': call_price,
            'put_price': put_price,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'current_price': current_price,
            'is_true_straddle': is_true_straddle,
            'expiry_date': formatted_expiry_date,
            'risk_free_rate': risk_free_rate,
            'call_iv': call_iv,
            'put_iv': put_iv
        }
        
        # Create the Black-Scholes pricing table for the current date
        bs_table = create_bs_pricing_table(bs_calculations, days_to_expiry)
        
        # Format the date display
        if days_to_expiry == 0:
            bs_date_display = html.Span("AT EXPIRATION", style={'fontWeight': 'bold', 'fontSize': '18px'})
        else:
            bs_date_display = f"{days_to_expiry} DAYS TO EXPIRY"
        
        # Calculate profit/loss for each price point
        profits = []
        for price in price_range:
            # At expiration
            if is_true_straddle:
                call_profit = max(0, price - strike_price) - call_price
                put_profit = max(0, strike_price - price) - put_price
            else:
                call_profit = max(0, price - call_strike_price) - call_price
                put_profit = max(0, put_strike_price - price) - put_price
                
            total_profit = call_profit + put_profit
            profits.append(total_profit)
        
        profit_df = pd.DataFrame({
            'Stock Price': price_range,
            'Profit/Loss': profits
        })
        
        # Create a more focused price range for better visualization
        # Zoom in to a more relevant range around the strikes and current price
        price_range_min = max(0.1, min(current_price * 0.8, lower_breakeven * 0.9))
        price_range_max = max(current_price * 1.2, upper_breakeven * 1.1)
        price_range = np.linspace(price_range_min, price_range_max, 100)
        
        # Recalculate profits for the zoomed-in range
        profits = []
        call_profits = []
        put_profits = []
        contract_profits = []
        
        for price in price_range:
            # At expiration
            if is_true_straddle:
                call_profit = max(0, price - strike_price) - call_price
                put_profit = max(0, strike_price - price) - put_price
            else:
                call_profit = max(0, price - call_strike_price) - call_price
                put_profit = max(0, put_strike_price - price) - put_price
                
            total_profit = call_profit + put_profit
            contract_profit = total_profit * 100
            
            profits.append(total_profit)
            call_profits.append(call_profit)
            put_profits.append(put_profit)
            contract_profits.append(contract_profit)
        
        profit_df = calculate_straddle_profit(
            current_price,
            call_strike,
            put_strike,
            call_price,
            put_price,
            price_range,
            is_true_straddle
        )
        
        # Create the profit/loss graph
        fig = go.Figure()
        
        # Add profit/loss line with improved hover template
        fig.add_trace(go.Scatter(
            x=profit_df['Stock Price'],
            y=profit_df['Profit/Loss'],
            mode='lines',
            name='Profit/Loss',
            line=dict(color=colors['accent'], width=3),
            hovertemplate='<b>Stock Price:</b> $%{x:.2f}<br>' +
                          '<b>P/L per Share:</b> $%{y:.2f}<br>' +
                          '<b>Call P/L:</b> $%{customdata[0]:.2f}<br>' +
                          '<b>Put P/L:</b> $%{customdata[1]:.2f}<br>' +
                          '<b>P/L per Contract:</b> $%{customdata[2]:.2f}<extra></extra>',
            customdata=np.column_stack((profit_df['Call P/L'], profit_df['Put P/L'], profit_df['Contract P/L']))
        ))
        
        # Calculate profit at current price
        current_price_profit = profit_df.loc[profit_df['Stock Price'].sub(current_price).abs().idxmin(), 'Profit/Loss']
        
        # Add breakeven points with annotations
        fig.add_trace(go.Scatter(
            x=[lower_breakeven, upper_breakeven],
            y=[0, 0],
            mode='markers+text',
            name='Breakeven Points',
            marker=dict(color=colors['text'], size=10, symbol='diamond'),
            text=["Lower BE", "Upper BE"],
            textposition="top center",
            hovertemplate='<b>Breakeven:</b> $%{x:.2f}<br><b>P/L:</b> $0.00<br><b>P/L per Contract:</b> $0.00<extra></extra>'
        ))
        
        # Calculate current price P/L components
        current_idx = profit_df['Stock Price'].sub(current_price).abs().idxmin()
        current_call_pl = profit_df.loc[current_idx, 'Call P/L']
        current_put_pl = profit_df.loc[current_idx, 'Put P/L']
        current_contract_pl = profit_df.loc[current_idx, 'Contract P/L']
        
        # Add current price marker with annotation
        fig.add_trace(go.Scatter(
            x=[current_price],
            y=[current_price_profit],
            mode='markers+text',
            name='Current Price',
            marker=dict(color=colors['accent'], size=12, symbol='circle'),
            text=["Current"],
            textposition="top center",
            hovertemplate='<b>Current Price:</b> $%{x:.2f}<br>' +
                          '<b>P/L per Share:</b> $%{y:.2f}<br>' +
                          '<b>Call P/L:</b> $' + f"{current_call_pl:.2f}" + '<br>' +
                          '<b>Put P/L:</b> $' + f"{current_put_pl:.2f}" + '<br>' +
                          '<b>P/L per Contract:</b> $' + f"{current_contract_pl:.2f}" + '<extra></extra>'
        ))
        
        # Add horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=price_range_min,
            y0=0,
            x1=price_range_max,
            y1=0,
            line=dict(color=colors['secondary'], width=2, dash="dash")
        )
        
        # Add vertical lines at strike prices with annotations
        if is_true_straddle:
            fig.add_shape(
                type="line",
                x0=strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['secondary'], width=2, dash="dash")
            )
            
            # Add annotation for strike price
            fig.add_annotation(
                x=strike_price,
                y=min(profit_df['Profit/Loss']),
                text=f"Strike: ${strike_price:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40,
                font=dict(color=colors['text'], size=12),
                bgcolor=colors['panel'],
                bordercolor=colors['secondary'],
                borderwidth=1,
                borderpad=4
            )
        else:
            # For strangle, add two vertical lines with annotations
            fig.add_shape(
                type="line",
                x0=call_strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=call_strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['profit'], width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=call_strike_price,
                y=min(profit_df['Profit/Loss']),
                text=f"Call Strike: ${call_strike_price:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40,
                font=dict(color=colors['profit'], size=12),
                bgcolor=colors['panel'],
                bordercolor=colors['secondary'],
                borderwidth=1,
                borderpad=4
            )
            
            fig.add_shape(
                type="line",
                x0=put_strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=put_strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['loss'], width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=put_strike_price,
                y=min(profit_df['Profit/Loss']),
                text=f"Put Strike: ${put_strike_price:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40,
                font=dict(color=colors['loss'], size=12),
                bgcolor=colors['panel'],
                bordercolor=colors['secondary'],
                borderwidth=1,
                borderpad=4
            )
        
        # Add annotations for breakeven points
        fig.add_annotation(
            x=lower_breakeven,
            y=0,
            text=f"Lower BE: ${lower_breakeven:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color=colors['text'], size=12),
            bgcolor=colors['panel'],
            bordercolor=colors['secondary'],
            borderwidth=1,
            borderpad=4
        )
        
        fig.add_annotation(
            x=upper_breakeven,
            y=0,
            text=f"Upper BE: ${upper_breakeven:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color=colors['text'], size=12),
            bgcolor=colors['panel'],
            bordercolor=colors['secondary'],
            borderwidth=1,
            borderpad=4
        )
        
        # Add max loss annotation
        fig.add_annotation(
            x=(lower_breakeven + upper_breakeven) / 2 if is_true_straddle else (put_strike_price + call_strike_price) / 2,
            y=-total_premium,
            text=f"Max Loss: -${total_premium:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=20,
            font=dict(color=colors['loss'], size=12),
            bgcolor=colors['panel'],
            bordercolor=colors['secondary'],
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout with improved settings
        fig.update_layout(
            title=f"{strategy_type} STRATEGY PROFIT/LOSS PROJECTION",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            legend=dict(
                bgcolor=colors['panel'],
                bordercolor=colors['secondary'],
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="closest",
            xaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid'],
                tickprefix="$",
                tickformat=".2f",
                showgrid=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor=colors['accent'],
                spikedash="solid",
                spikethickness=1
            ),
            yaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid'],
                tickprefix="$",
                tickformat=".2f",
                showgrid=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor=colors['accent'],
                spikedash="solid",
                spikethickness=1
            ),
            # Add a range slider for better navigation
            xaxis_rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor=colors['panel']
            )
        )
        
        # Prepare strategy results
        strategy_results = html.Div([
            html.H4(f"{strategy_type} STRATEGY DETAILS", style={'color': colors['accent'], 'marginTop': '0', 'textAlign': 'center'}),
            
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5("CALL OPTION", style={'color': colors['profit'], 'marginTop': '0'}),
                    html.P(f"Price: ${call_price:.2f}", style={'margin': '5px 0'}),
                    html.P(f"Strike: ${call_strike:.2f}", style={'margin': '5px 0'}),
                    html.P(f"IV: {call_data['IV']}%", style={'margin': '5px 0'}),
                ]),
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5("PUT OPTION", style={'color': colors['loss'], 'marginTop': '0'}),
                    html.P(f"Price: ${put_price:.2f}", style={'margin': '5px 0'}),
                    html.P(f"Strike: ${put_strike:.2f}", style={'margin': '5px 0'}),
                    html.P(f"IV: {put_data['IV']}%", style={'margin': '5px 0'}),
                ]),
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5(f"{strategy_type} COST", style={'color': colors['accent'], 'marginTop': '0'}),
                    html.P(f"Total Premium: ${total_premium:.2f}", style={'margin': '5px 0'}),
                    html.Div([
                        html.Span("Straddle Cost: ", style={'fontWeight': 'bold', 'verticalAlign': 'middle'}),
                        html.Span(f"${total_premium * 100:.2f}", style={
                            'color': colors['loss'], 
                            'fontWeight': 'bold',
                            'verticalAlign': 'middle',
                            'display': 'inline-block'
                        })
                    ], style={'margin': '5px 0', 'lineHeight': '1.5'}),
                ]),
            ]),
            
            html.Div(style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '5px', 'marginTop': '15px'}, children=[
                html.H5("BREAKEVEN ANALYSIS", style={'color': colors['accent'], 'marginTop': '0'}),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}, children=[
                    html.Div([
                        html.P("Lower Breakeven:", style={'margin': '5px 0'}),
                        html.P(f"${lower_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((lower_breakeven / current_price) - 1) * 100:.2f}% from current)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['loss']}),
                    ]),
                    html.Div([
                        html.P("Upper Breakeven:", style={'margin': '5px 0'}),
                        html.P(f"${upper_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((upper_breakeven / current_price) - 1) * 100:.2f}% from current)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['profit']}),
                    ]),
                    html.Div([
                        html.P("Breakeven Range:", style={'margin': '5px 0'}),
                        html.P(f"${upper_breakeven - lower_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((upper_breakeven - lower_breakeven) / current_price) * 100:.2f}% of current price)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['text']}),
                    ]),
                ]),
            ]),
            
            html.Div(style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '5px', 'marginTop': '15px'}, children=[
                html.H5("MAX PROFIT/LOSS POTENTIAL", style={'color': colors['accent'], 'marginTop': '0'}),
                html.P("Maximum Loss: Limited to total premium paid", style={'margin': '5px 0'}),
                html.P(f"${total_premium:.2f} per share (${total_premium * 100:.2f} for straddle)", 
                       style={'margin': '5px 0', 'color': colors['loss'], 'fontWeight': 'bold'}),
                html.P("Maximum Profit: Unlimited as stock price moves away from strike", style={'margin': '5px 0'}),
                html.P("Profit increases as price moves further from strike in either direction", 
                       style={'margin': '5px 0', 'color': colors['profit'], 'fontWeight': 'bold'}),
            ]),
        ])
        
        return strategy_results, fig, bs_calculations, bs_date_display, bs_table, json.dumps(price_scenarios.tolist())
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"Error in update_results: {error_message}")
        return (
            html.P(error_message, style={'color': colors['loss']}),
            go.Figure(),
            None,
            "Error",
            html.Div(error_message, style={'color': colors['loss']}),
            None
        )

# Function to create Black-Scholes pricing table
def create_bs_pricing_table(bs_calculations, days):
    if not bs_calculations or days not in bs_calculations['days_list']:
        return html.Div("No data available")
    
    # Get the data for the selected days
    data = bs_calculations['results'][days]
    call_price = bs_calculations['call_price']
    put_price = bs_calculations['put_price']
    current_price = bs_calculations['current_price']
    is_true_straddle = bs_calculations['is_true_straddle']
    expiry_date = bs_calculations.get('expiry_date')
    
    # Sort data by stock price in descending order (high to low)
    data = sorted(data, key=lambda x: x['Stock Price'], reverse=True)
    
    # Create the table header
    header = html.Thead(html.Tr([
        html.Th("Stock Price", style={'backgroundColor': colors['secondary'], 'color': colors['text'], 'padding': '10px', 'textAlign': 'center'}),
        html.Th("Call Contract Value", style={'backgroundColor': colors['secondary'], 'color': colors['text'], 'padding': '10px', 'textAlign': 'center'}),
        html.Th("Put Contract Value", style={'backgroundColor': colors['secondary'], 'color': colors['text'], 'padding': '10px', 'textAlign': 'center'}),
        html.Th(f"Total Premium (${(call_price + put_price) * 100:.2f})", style={'backgroundColor': colors['secondary'], 'color': colors['text'], 'padding': '10px', 'textAlign': 'center'}),
        html.Th("Contract P/L", style={'backgroundColor': colors['secondary'], 'color': colors['text'], 'padding': '10px', 'textAlign': 'center'})
    ]))
    
    # Get breakeven points using the same calculation as in strategy details
    lower_breakeven, upper_breakeven = calculate_breakeven_points(
        bs_calculations['call_strike'],
        bs_calculations['put_strike'],
        call_price,
        put_price,
        is_true_straddle
    )
    
    # Create the table rows
    rows = []
    
    # First add the breakeven points explicitly if they're not in the data
    be_points = [lower_breakeven, upper_breakeven]
    for be_price in be_points:
        if not any(abs(item['Stock Price'] - be_price) < 0.01 for item in data):
            # Calculate values at breakeven
            call_value = black_scholes(
                be_price,
                bs_calculations['call_strike'],
                days/365,
                bs_calculations['risk_free_rate']/100,
                bs_calculations['call_iv']/100,
                "call"
            )
            put_value = black_scholes(
                be_price,
                bs_calculations['put_strike'],
                days/365,
                bs_calculations['risk_free_rate']/100,
                bs_calculations['put_iv']/100,
                "put"
            )
            data.append({
                'Stock Price': be_price,
                'Call Price': call_value,
                'Put Price': put_value
            })
    
    # Re-sort the data by stock price in descending order
    data = sorted(data, key=lambda x: x['Stock Price'], reverse=True)
    
    for item in data:
        stock_price = item['Stock Price']
        call_value = item['Call Price']
        put_value = item['Put Price']
        
        # Calculate values and P/L
        call_contract_value = call_value * 100
        put_contract_value = put_value * 100
        total_contract_value = call_contract_value + put_contract_value
        total_premium = (call_price + put_price) * 100
        contract_pl = total_contract_value - total_premium
        
        # Determine row style based on price type
        is_breakeven = abs(contract_pl) < 1.0  # Contract P/L is close to zero (breakeven point)
        is_current = abs(stock_price - current_price) < 0.01
        is_exact_be = any(abs(stock_price - be_price) < 0.01 for be_price in [lower_breakeven, upper_breakeven])
        
        if is_exact_be or is_breakeven:
            # Breakeven point - highlight in a distinct way
            row_style = {
                'backgroundColor': colors['secondary'], 
                'color': colors['text'], 
                'fontWeight': 'bold', 
                'borderTop': f'2px solid {colors["accent"]}', 
                'borderBottom': f'2px solid {colors["accent"]}'
            }
            # Add breakeven label to stock price
            if is_exact_be:
                be_type = "Lower BE" if abs(stock_price - lower_breakeven) < 0.01 else "Upper BE"
                stock_price_display = f"${stock_price:.2f} ({be_type})"
            else:
                stock_price_display = f"${stock_price:.2f} (Breakeven)"
        elif is_current:
            # Current price - white background with black text
            row_style = {'backgroundColor': '#ffffff', 'color': '#000000', 'fontWeight': 'bold'}
        elif abs(stock_price - current_price) < current_price * 0.02:  # Within 2% of current price
            row_style = {'backgroundColor': colors['panel'], 'color': colors['text']}
        else:
            row_style = {}
        
        # Determine cell styles - use red for 0 or negative values, white for positive values
        call_value_style = {'color': colors['loss'] if call_value <= 0 else colors['text'], 'fontWeight': 'bold'}
        put_value_style = {'color': colors['loss'] if put_value <= 0 else colors['text'], 'fontWeight': 'bold'}
        call_contract_style = {'color': colors['loss'] if call_contract_value <= 0 else colors['text'], 'fontWeight': 'bold'}
        put_contract_style = {'color': colors['loss'] if put_contract_value <= 0 else colors['text'], 'fontWeight': 'bold'}
        total_contract_style = {'color': colors['profit'] if total_contract_value > total_premium else colors['loss'], 'fontWeight': 'bold'}
        contract_pl_style = {'color': colors['profit'] if contract_pl > 0 else colors['loss'], 'fontWeight': 'bold'}
        
        # Create row with proper styling that ensures contract values are properly colored
        # Ensure 0 or negative values are always red
        row = html.Tr([
            html.Td(stock_price_display if is_breakeven else f"${stock_price:.2f}", 
                   style={'padding': '8px', 'textAlign': 'center', **row_style}),
            # Force call contract value to use red for 0 or negative values, green for positive
            html.Td(f"${call_contract_value:.2f}", style={
                'padding': '8px', 
                'textAlign': 'center',
                'color': colors['loss'] if call_contract_value <= 0 or abs(call_contract_value) < 0.001 else colors['profit'],
                'fontWeight': 'bold',
                **{k: v for k, v in row_style.items() if k not in ['color', 'fontWeight']}
            }),
            # Force put contract value to use red for 0 or negative values, green for positive
            html.Td(f"${put_contract_value:.2f}", style={
                'padding': '8px', 
                'textAlign': 'center',
                'color': colors['loss'] if put_contract_value <= 0 or abs(put_contract_value) < 0.001 else colors['profit'],
                'fontWeight': 'bold',
                **{k: v for k, v in row_style.items() if k not in ['color', 'fontWeight']}
            }),
            html.Td(f"${total_premium:.2f}", style={'padding': '8px', 'textAlign': 'center', **row_style}),
            # Force contract P/L to use the contract_pl_style regardless of row_style
            html.Td(f"${contract_pl:.2f}", style={
                'padding': '8px', 
                'textAlign': 'center',
                'color': colors['loss'] if contract_pl <= 0 or abs(contract_pl) < 0.001 else contract_pl_style['color'],
                'fontWeight': 'bold',
                **{k: v for k, v in row_style.items() if k not in ['color', 'fontWeight']}
            })
        ])
        rows.append(row)
    
    # Create the table body
    body = html.Tbody(rows)
    
    # Create the table with a title showing the current price and expiry information
    table_container = html.Div([
        html.Div([
            html.Div([
                html.Span("Current Price: ", style={'fontWeight': 'bold'}),
                html.Span(f"${current_price:.2f}", style={
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '5px'
                }),
            ], style={'marginBottom': '10px', 'textAlign': 'center'}),
            
            html.Div([
                html.Span("Total Premium Paid: ", style={'fontWeight': 'bold'}),
                html.Span(f"${call_price + put_price:.2f}/share (${(call_price + put_price) * 100:.2f}/contract)", style={
                    'backgroundColor': colors['loss'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '5px'
                }),
            ], style={'marginBottom': '10px', 'textAlign': 'center'}),
            
            html.Div([
                html.Span("Expiration Date: ", style={'fontWeight': 'bold'}),
                html.Span(expiry_date if expiry_date else "N/A", style={
                    'backgroundColor': colors['secondary'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '5px'
                }),
            ], style={'marginBottom': '10px', 'textAlign': 'center'}),
            
            html.Div([
                html.Span("Days to Expiry: ", style={'fontWeight': 'bold'}),
                html.Span(f"{days}", style={
                    'backgroundColor': colors['secondary'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '5px'
                }),
            ], style={'marginBottom': '10px', 'textAlign': 'center'}),
            
            html.Div([
                html.Span("Breakeven Points: ", style={'fontWeight': 'bold'}),
                html.Span(f"Lower: ${lower_breakeven:.2f}", style={
                    'backgroundColor': colors['secondary'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '5px'
                }),
                html.Span(f"Upper: ${upper_breakeven:.2f}", style={
                    'backgroundColor': colors['secondary'],
                    'color': colors['text'],
                    'padding': '3px 8px',
                    'borderRadius': '3px',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '15px', 'textAlign': 'center'})
        ]),
        
        html.Table([header, body], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'border': f'1px solid {colors["secondary"]}',
            'backgroundColor': colors['background']
        })
    ])
    
    return table_container

# Callback for Black-Scholes date navigation (previous date)
@app.callback(
    [Output('bs-date-display', 'children', allow_duplicate=True),
     Output('bs-pricing-table', 'children', allow_duplicate=True)],
    [Input('prev-date-button', 'n_clicks')],
    [State('bs-calculations', 'data'),
     State('bs-date-display', 'children')],
    prevent_initial_call=True
)
def navigate_to_prev_date(n_clicks, bs_calculations, current_display):
    if not bs_calculations or not n_clicks:
        return dash.no_update, dash.no_update
    
    try:
        # Get current days
        current_days = bs_calculations['current_days']
        days_list = bs_calculations['days_list']
        
        # Find the previous date in the list
        current_index = days_list.index(current_days)
        if current_index > 0:
            new_days = days_list[current_index - 1]
        else:
            # Already at the earliest date
            return dash.no_update, dash.no_update
        
        # Update the display
        if new_days == 0:
            new_display = "AT EXPIRATION"
        else:
            new_display = f"{new_days} DAYS TO EXPIRY"
        
        # Create the new table
        new_table = create_bs_pricing_table(bs_calculations, new_days)
        
        # Update the current days in the store
        bs_calculations['current_days'] = new_days
        
        return new_display, new_table
        
    except Exception as e:
        print(f"Error in navigate_to_prev_date: {str(e)}")
        return dash.no_update, dash.no_update

# Callback for Black-Scholes date navigation (next date)
@app.callback(
    [Output('bs-date-display', 'children', allow_duplicate=True),
     Output('bs-pricing-table', 'children', allow_duplicate=True)],
    [Input('next-date-button', 'n_clicks')],
    [State('bs-calculations', 'data'),
     State('bs-date-display', 'children')],
    prevent_initial_call=True
)
def navigate_to_next_date(n_clicks, bs_calculations, current_display):
    if not bs_calculations or not n_clicks:
        return dash.no_update, dash.no_update
    
    try:
        # Get current days
        current_days = bs_calculations['current_days']
        days_list = bs_calculations['days_list']
        
        # Find the next date in the list
        current_index = days_list.index(current_days)
        if current_index < len(days_list) - 1:
            new_days = days_list[current_index + 1]
        else:
            # Already at the latest date
            return dash.no_update, dash.no_update
        
        # Update the display
        if new_days == 0:
            new_display = "AT EXPIRATION"
        else:
            new_display = f"{new_days} DAYS TO EXPIRY"
        
        # Create the new table
        new_table = create_bs_pricing_table(bs_calculations, new_days)
        
        # Update the current days in the store
        bs_calculations['current_days'] = new_days
        
        return new_display, new_table
        
    except Exception as e:
        print(f"Error in navigate_to_next_date: {str(e)}")
        return dash.no_update, dash.no_update

# Callback for previous expiration button
@app.callback(
    [Output('calls-table-container', 'children', allow_duplicate=True),
     Output('puts-table-container', 'children', allow_duplicate=True),
     Output('expiration-display', 'children', allow_duplicate=True),
     Output('prev-expiry', 'data', allow_duplicate=True),
     Output('current-expiry', 'data', allow_duplicate=True),
     Output('next-expiry', 'data', allow_duplicate=True),
     Output('selected-call', 'data', allow_duplicate=True),
     Output('selected-put', 'data', allow_duplicate=True)],
    [Input('prev-expiry-button', 'n_clicks')],
    [State('prev-expiry', 'data'),
     State('current-ticker', 'data'),
     State('all-expiries', 'data')],
    prevent_initial_call=True
)
def go_to_prev_expiration(n_clicks, prev_exp, ticker, all_expiries_json):
    if not prev_exp or not ticker:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        all_expiries = json.loads(all_expiries_json)
        current_index = all_expiries.index(prev_exp)
        
        # Calculate new navigation indices
        new_prev = all_expiries[current_index - 1] if current_index > 0 else None
        new_next = all_expiries[current_index + 1] if current_index < len(all_expiries) - 1 else None
        
        # Get options chain for the selected expiration
        calls_df, puts_df, _, _, current_price, exp_date_formatted, _, _ = get_options_chain(ticker, prev_exp)
        
        if calls_df is None or puts_df is None:
            return (
                html.Div("No options data available for this date", style={'color': colors['loss']}),
                html.Div("No options data available for this date", style={'color': colors['loss']}),
                "No data available",
                new_prev,
                prev_exp,
                new_next,
                None,
                None
            )
        
        # Find the closest strikes to current price
        closest_call_idx = (calls_df['Strike'] - current_price).abs().idxmin()
        closest_put_idx = (puts_df['Strike'] - current_price).abs().idxmin()
        
        # Sort the dataframes by strike price
        calls_df = calls_df.sort_values('Strike')
        puts_df = puts_df.sort_values('Strike')
        
        # Get 5 strikes above and below the closest strike
        call_start_idx = max(0, closest_call_idx - 5)
        call_end_idx = min(len(calls_df) - 1, closest_call_idx + 5)
        put_start_idx = max(0, closest_put_idx - 5)
        put_end_idx = min(len(puts_df) - 1, closest_put_idx + 5)
        
        # Filter the dataframes to show options around current price
        visible_calls_df = calls_df.iloc[call_start_idx:call_end_idx + 1].copy()
        visible_puts_df = puts_df.iloc[put_start_idx:put_end_idx + 1].copy()
        
        # Add a column to highlight the row closest to current price
        visible_calls_df['Near Current'] = False
        closest_visible_call_idx = (visible_calls_df['Strike'] - current_price).abs().idxmin()
        visible_calls_df.loc[closest_visible_call_idx, 'Near Current'] = True
        
        visible_puts_df['Near Current'] = False
        closest_visible_put_idx = (visible_puts_df['Strike'] - current_price).abs().idxmin()
        visible_puts_df.loc[closest_visible_put_idx, 'Near Current'] = True
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in visible_calls_df.columns if col != 'Near Current'
            ],
            data=visible_calls_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(0, 255, 127, 0.2)',
                    'color': colors['profit']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in puts_df.columns
            ],
            data=visible_puts_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(255, 71, 87, 0.2)',
                    'color': colors['loss']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        return calls_table, puts_table, exp_date_formatted, new_prev, prev_exp, new_next, None, None
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            html.Div(error_message, style={'color': colors['loss']}),
            html.Div(error_message, style={'color': colors['loss']}),
            None,
            None
        )

# Callback for next expiration button
@app.callback(
    [Output('calls-table-container', 'children', allow_duplicate=True),
     Output('puts-table-container', 'children', allow_duplicate=True),
     Output('expiration-display', 'children', allow_duplicate=True),
     Output('prev-expiry', 'data', allow_duplicate=True),
     Output('current-expiry', 'data', allow_duplicate=True),
     Output('next-expiry', 'data', allow_duplicate=True),
     Output('selected-call', 'data', allow_duplicate=True),
     Output('selected-put', 'data', allow_duplicate=True)],
    [Input('next-expiry-button', 'n_clicks')],
    [State('next-expiry', 'data'),
     State('current-ticker', 'data'),
     State('all-expiries', 'data')],
    prevent_initial_call=True
)
def go_to_next_expiration(n_clicks, next_exp, ticker, all_expiries_json):
    if not next_exp or not ticker:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        all_expiries = json.loads(all_expiries_json)
        current_index = all_expiries.index(next_exp)
        
        # Calculate new navigation indices
        new_prev = all_expiries[current_index - 1] if current_index > 0 else None
        new_next = all_expiries[current_index + 1] if current_index < len(all_expiries) - 1 else None
        
        # Get options chain for the selected expiration
        calls_df, puts_df, _, _, current_price, exp_date_formatted, _, _ = get_options_chain(ticker, next_exp)
        
        if calls_df is None or puts_df is None:
            return (
                html.Div("No options data available for this date", style={'color': colors['loss']}),
                html.Div("No options data available for this date", style={'color': colors['loss']}),
                "No data available",
                new_prev,
                next_exp,
                new_next,
                None,
                None
            )
        
        # Find the closest strikes to current price
        closest_call_idx = (calls_df['Strike'] - current_price).abs().idxmin()
        closest_put_idx = (puts_df['Strike'] - current_price).abs().idxmin()
        
        # Sort the dataframes by strike price
        calls_df = calls_df.sort_values('Strike')
        puts_df = puts_df.sort_values('Strike')
        
        # Get 5 strikes above and below the closest strike
        call_start_idx = max(0, closest_call_idx - 5)
        call_end_idx = min(len(calls_df) - 1, closest_call_idx + 5)
        put_start_idx = max(0, closest_put_idx - 5)
        put_end_idx = min(len(puts_df) - 1, closest_put_idx + 5)
        
        # Filter the dataframes to show options around current price
        visible_calls_df = calls_df.iloc[call_start_idx:call_end_idx + 1].copy()
        visible_puts_df = puts_df.iloc[put_start_idx:put_end_idx + 1].copy()
        
        # Add a column to highlight the row closest to current price
        visible_calls_df['Near Current'] = False
        closest_visible_call_idx = (visible_calls_df['Strike'] - current_price).abs().idxmin()
        visible_calls_df.loc[closest_visible_call_idx, 'Near Current'] = True
        
        visible_puts_df['Near Current'] = False
        closest_visible_put_idx = (visible_puts_df['Strike'] - current_price).abs().idxmin()
        visible_puts_df.loc[closest_visible_put_idx, 'Near Current'] = True
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in visible_calls_df.columns if col != 'Near Current'
            ],
            data=visible_calls_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(0, 255, 127, 0.2)',
                    'color': colors['profit']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in visible_puts_df.columns if col != 'Near Current'
            ],
            data=visible_puts_df.to_dict('records'),
            style_header={
                'backgroundColor': colors['secondary'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '70px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'ITM', 'filter_query': '{ITM} eq True'},
                    'backgroundColor': 'rgba(255, 71, 87, 0.2)',
                    'color': colors['loss']
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': colors['accent'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["text"]}'
                },
                {
                    'if': {'filter_query': '{Near Current} eq true'},
                    'backgroundColor': colors['secondary'],
                    'fontWeight': 'bold'
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_action='none',  # No pagination to show all visible options
            style_table={'height': '400px', 'overflowY': 'auto'}
        )
        
        return calls_table, puts_table, exp_date_formatted, new_prev, next_exp, new_next, None, None
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            html.Div(error_message, style={'color': colors['loss']}),
            html.Div(error_message, style={'color': colors['loss']}),
            error_message,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            None,
            None
        )

# Run the app
if __name__ == '__main__':
    # Add a startup message to help with troubleshooting
    print("Starting Ultron Straddle Strategy Analyzer...")
    print("Using yfinance version:", yf.__version__)
    print("Make sure you have an active internet connection to fetch options data")
    app.run_server(debug=True)
