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
    r: Risk-free interest rate
    sigma: Volatility
    option_type: "call" or "put"
    
    Returns:
    Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Calculate straddle strategy profit at different stock prices
def calculate_straddle_profit(current_price, strike_price, call_price, put_price, price_range):
    """
    Calculate profit/loss for a straddle strategy at different stock prices
    
    Parameters:
    current_price: Current stock price
    strike_price: Strike price for both call and put
    call_price: Premium paid for call option
    put_price: Premium paid for put option
    price_range: Array of stock prices to calculate profit/loss
    
    Returns:
    DataFrame with stock prices and corresponding profit/loss
    """
    total_premium = call_price + put_price
    
    profits = []
    for price in price_range:
        # At expiration
        call_profit = max(0, price - strike_price) - call_price
        put_profit = max(0, strike_price - price) - put_price
        total_profit = call_profit + put_profit
        profits.append(total_profit)
    
    return pd.DataFrame({
        'Stock Price': price_range,
        'Profit/Loss': profits
    })

# Calculate breakeven points for straddle
def calculate_breakeven_points(strike_price, total_premium):
    """
    Calculate breakeven points for a straddle strategy
    
    Parameters:
    strike_price: Strike price for both options
    total_premium: Total premium paid for both options
    
    Returns:
    Tuple of lower and upper breakeven points
    """
    lower_breakeven = strike_price - total_premium
    upper_breakeven = strike_price + total_premium
    
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
        # Create ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = ticker_obj.options
        
        if not expirations or len(expirations) == 0:
            raise ValueError(f"No options data available for {ticker}")
        
        # Use selected expiration or default to first available
        exp_date = selected_expiration if selected_expiration in expirations else expirations[0]
        
        # Get options for this expiration
        options = ticker_obj.option_chain(exp_date)
        
        # Get current price
        current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
        
        # Format the data
        calls = options.calls
        puts = options.puts
        
        if calls.empty or puts.empty:
            raise ValueError(f"No options chain data available for {ticker} on {exp_date}")
        
        # Format the data
        calls = calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'inTheMoney']]
        calls.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'IV', 'ITM']
        calls['IV'] = (calls['IV'] * 100).round(2)
        
        puts = puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'inTheMoney']]
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
        return None, None, None, None, None, None, None, None

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    # Header
    html.Div(style={'padding': '20px', 'textAlign': 'center', 'borderBottom': f'1px solid {colors["secondary"]}'}, children=[
        html.H1("ULTRON STRADDLE STRATEGY ANALYZER", style={'color': colors['accent'], 'fontWeight': 'bold', 'letterSpacing': '2px'}),
        html.H3("Advanced Options Trading Intelligence System", style={'color': colors['text'], 'fontStyle': 'italic'})
    ]),
    
    # Store components for selected options
    dcc.Store(id='selected-call', storage_type='memory'),
    dcc.Store(id='selected-put', storage_type='memory'),
    dcc.Store(id='stock-price-store', storage_type='memory'),
    
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
            
            # Profit/Loss graph
            dcc.Graph(
                id='profit-loss-graph',
                config={'displayModeBar': False},
                style={'backgroundColor': colors['background']}
            )
        ]),
    ]),
    
    # Footer
    html.Div(style={'padding': '20px', 'textAlign': 'center', 'borderTop': f'1px solid {colors["secondary"]}'}, children=[
        html.P("ULTRON FINANCIAL SYSTEMS © 2025", style={'color': colors['secondary']}),
        html.P("DISCLAIMER: This tool is for educational purposes only. Not financial advice.", style={'color': colors['secondary'], 'fontSize': '12px'})
    ])
])

# Store for current ticker
dcc.Store(id='current-ticker', storage_type='memory'),

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
        # Normalize ticker
        ticker = ticker.strip().upper()
        
        # Get options chain with simplified approach
        calls_df, puts_df, exp_date, all_expirations, current_price, exp_date_formatted, prev_exp, next_exp = get_options_chain(ticker)
        
        if calls_df is None or puts_df is None:
            return (
                html.Div("No options data available", style={'color': colors['loss']}),
                html.Div("No options data available", style={'color': colors['loss']}),
                html.Div([
                    html.H4(f"{ticker}", style={'color': colors['accent'], 'marginTop': '0'}),
                    html.P("Could not retrieve options data", style={'margin': '5px 0', 'color': colors['loss']})
                ]),
                "No expiration dates available",
                None,
                None,
                None,
                None,
                None,
                None,
                None
            )
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in calls_df.columns
            ],
            data=calls_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in puts_df.columns
            ],
            data=puts_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        # Stock info display
        stock_info = html.Div([
            html.H4(f"{ticker}", style={'color': colors['accent'], 'marginTop': '0'}),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                html.Div([
                    html.P("CURRENT PRICE:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"${current_price:.2f}", style={'margin': '5px 0', 'fontSize': '18px'})
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

# Callback for strategy analysis
@app.callback(
    [Output('strategy-results', 'children'),
     Output('profit-loss-graph', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('selected-call', 'data'),
     State('selected-put', 'data'),
     State('stock-price-store', 'data'),
     State('risk-free-rate', 'value')]
)
def update_results(n_clicks, call_data, put_data, stock_price_data, risk_free_rate):
    if n_clicks == 0 or not call_data or not put_data or not stock_price_data:
        # Initial state or missing data
        return (
            html.P("Select both a call and put option, then click ANALYZE STRATEGY", style={'color': colors['secondary']}),
            go.Figure()
        )
    
    try:
        # Get data from selected options
        call_price = call_data['Last Price']
        put_price = put_data['Last Price']
        call_strike = call_data['Strike']
        put_strike = put_data['Strike']
        current_price = json.loads(stock_price_data)['price']
        
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
        
        # Calculate breakeven points
        if is_true_straddle:
            lower_breakeven, upper_breakeven = calculate_breakeven_points(strike_price, total_premium)
        else:
            # For a strangle, breakeven points are different
            lower_breakeven = put_strike_price - put_price
            upper_breakeven = call_strike_price + call_price
        
        # Calculate profit/loss at different stock prices
        price_range_min = max(0.1, current_price * 0.5)  # Avoid negative or zero prices
        price_range_max = current_price * 1.5
        price_range = np.linspace(price_range_min, price_range_max, 100)
        
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
        
        # Create the profit/loss graph
        fig = go.Figure()
        
        # Add profit/loss line
        fig.add_trace(go.Scatter(
            x=profit_df['Stock Price'],
            y=profit_df['Profit/Loss'],
            mode='lines',
            name='Profit/Loss',
            line=dict(color=colors['accent'], width=3)
        ))
        
        # Add breakeven points
        fig.add_trace(go.Scatter(
            x=[lower_breakeven, upper_breakeven],
            y=[0, 0],
            mode='markers',
            name='Breakeven Points',
            marker=dict(color=colors['text'], size=10, symbol='diamond')
        ))
        
        # Add current price marker
        fig.add_trace(go.Scatter(
            x=[current_price],
            y=[profit_df.loc[profit_df['Stock Price'].sub(current_price).abs().idxmin(), 'Profit/Loss']],
            mode='markers',
            name='Current Price',
            marker=dict(color=colors['accent'], size=12, symbol='circle')
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
        
        # Add vertical lines at strike prices
        if is_true_straddle:
            fig.add_shape(
                type="line",
                x0=strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['secondary'], width=2, dash="dash")
            )
        else:
            # For strangle, add two vertical lines
            fig.add_shape(
                type="line",
                x0=call_strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=call_strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['profit'], width=2, dash="dash")
            )
            fig.add_shape(
                type="line",
                x0=put_strike_price,
                y0=min(profit_df['Profit/Loss']),
                x1=put_strike_price,
                y1=max(profit_df['Profit/Loss']),
                line=dict(color=colors['loss'], width=2, dash="dash")
            )
        
        # Update layout
        fig.update_layout(
            title=f"{strategy_type} STRATEGY PROFIT/LOSS PROJECTION",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            legend=dict(
                bgcolor=colors['panel'],
                bordercolor=colors['secondary']
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x unified",
            xaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid']
            ),
            yaxis=dict(
                gridcolor=colors['grid'],
                zerolinecolor=colors['grid']
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
                    html.P(f"Per Contract: ${total_premium * 100:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold'}),
                ]),
            ]),
            
            html.Div(style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '5px', 'marginTop': '15px'}, children=[
                html.H5("BREAKEVEN ANALYSIS", style={'color': colors['accent'], 'marginTop': '0'}),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}, children=[
                    html.Div([
                        html.P("Lower Breakeven:", style={'margin': '5px 0'}),
                        html.P(f"${lower_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((lower_breakeven / current_price) - 1) * 100:.2f}% from current)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['secondary']}),
                    ]),
                    html.Div([
                        html.P("Upper Breakeven:", style={'margin': '5px 0'}),
                        html.P(f"${upper_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((upper_breakeven / current_price) - 1) * 100:.2f}% from current)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['secondary']}),
                    ]),
                    html.Div([
                        html.P("Breakeven Range:", style={'margin': '5px 0'}),
                        html.P(f"${upper_breakeven - lower_breakeven:.2f}", style={'margin': '5px 0', 'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.P(f"({((upper_breakeven - lower_breakeven) / current_price) * 100:.2f}% of current price)", style={'margin': '5px 0', 'fontSize': '12px', 'color': colors['secondary']}),
                    ]),
                ]),
            ]),
            
            html.Div(style={'backgroundColor': colors['panel'], 'padding': '15px', 'borderRadius': '5px', 'marginTop': '15px'}, children=[
                html.H5("MAX PROFIT/LOSS POTENTIAL", style={'color': colors['accent'], 'marginTop': '0'}),
                html.P("Maximum Loss: Limited to total premium paid", style={'margin': '5px 0'}),
                html.P(f"${total_premium:.2f} per share (${total_premium * 100:.2f} per contract)", 
                       style={'margin': '5px 0', 'color': colors['loss'], 'fontWeight': 'bold'}),
                html.P("Maximum Profit: Unlimited as stock price moves away from strike", style={'margin': '5px 0'}),
                html.P("Profit increases as price moves further from strike in either direction", 
                       style={'margin': '5px 0', 'color': colors['profit'], 'fontWeight': 'bold'}),
            ]),
        ])
        
        return strategy_results, fig
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            html.P(error_message, style={'color': colors['loss']}),
            go.Figure()
        )

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
        calls_df, puts_df, _, _, _, exp_date_formatted, _, _ = get_options_chain(ticker, prev_exp)
        
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
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in calls_df.columns
            ],
            data=calls_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in puts_df.columns
            ],
            data=puts_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
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
        calls_df, puts_df, _, _, _, exp_date_formatted, _, _ = get_options_chain(ticker, next_exp)
        
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
        
        # Create interactive tables
        calls_table = dash_table.DataTable(
            id='calls-table',
            columns=[
                {"name": col, "id": col} for col in calls_df.columns
            ],
            data=calls_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        puts_table = dash_table.DataTable(
            id='puts-table',
            columns=[
                {"name": col, "id": col} for col in puts_df.columns
            ],
            data=puts_df.to_dict('records'),
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
                }
            ],
            row_selectable='single',
            selected_rows=[],
            page_size=10
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
    app.run_server(debug=True)
