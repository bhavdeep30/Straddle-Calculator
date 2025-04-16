import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime

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

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    # Header
    html.Div(style={'padding': '20px', 'textAlign': 'center', 'borderBottom': f'1px solid {colors["secondary"]}'}, children=[
        html.H1("ULTRON STRADDLE STRATEGY ANALYZER", style={'color': colors['accent'], 'fontWeight': 'bold', 'letterSpacing': '2px'}),
        html.H3("Advanced Options Trading Intelligence System", style={'color': colors['text'], 'fontStyle': 'italic'})
    ]),
    
    # Main content
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
        # Left panel - Inputs
        html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, children=[
            html.H3("CONTROL PARAMETERS", style={'color': colors['accent'], 'borderBottom': f'1px solid {colors["secondary"]}', 'paddingBottom': '10px'}),
            
            # Ticker input
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("TICKER SYMBOL", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(
                    id='ticker-input',
                    type='text',
                    value='AAPL',
                    style={'width': '100%', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px'}
                ),
            ]),
            
            # Strike price input
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("STRIKE PRICE ($)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(
                    id='strike-price-input',
                    type='number',
                    value=150,
                    style={'width': '100%', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px'}
                ),
            ]),
            
            # Days to expiration
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("DAYS TO EXPIRATION", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(
                    id='days-to-expiration',
                    type='number',
                    value=30,
                    min=1,
                    style={'width': '100%', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px'}
                ),
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
            
            # Volatility override
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("CUSTOM VOLATILITY (%)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                    dcc.Input(
                        id='volatility-override',
                        type='number',
                        value=30,
                        min=1,
                        max=200,
                        step=0.1,
                        style={'flex': '1', 'backgroundColor': colors['background'], 'color': colors['text'], 'border': f'1px solid {colors["secondary"]}', 'padding': '10px', 'borderRadius': '5px'}
                    ),
                    html.Div(style={'marginLeft': '10px'}, children=[
                        dcc.Checklist(
                            id='use-custom-volatility',
                            options=[{'label': 'USE CUSTOM', 'value': 'yes'}],
                            value=[],
                            style={'color': colors['text']}
                        )
                    ])
                ]),
            ]),
            
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
                    'marginTop': '10px'
                }
            ),
            
            # Stock info display
            html.Div(id='stock-info', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '5px', 'border': f'1px solid {colors["secondary"]}'}),
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

@app.callback(
    [Output('stock-info', 'children'),
     Output('strategy-results', 'children'),
     Output('profit-loss-graph', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('strike-price-input', 'value'),
     State('days-to-expiration', 'value'),
     State('risk-free-rate', 'value'),
     State('volatility-override', 'value'),
     State('use-custom-volatility', 'value')]
)
def update_results(n_clicks, ticker, strike_price, days_to_expiration, risk_free_rate, volatility_override, use_custom_volatility):
    if n_clicks == 0:
        # Initial state
        return (
            html.P("Enter parameters and click ANALYZE STRATEGY to begin", style={'color': colors['secondary']}),
            html.P("Strategy analysis will appear here", style={'color': colors['secondary']}),
            go.Figure()
        )
    
    # Fetch stock data
    try:
        stock = yf.Ticker(ticker)
        current_data = stock.history(period="1d")
        if current_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        current_price = current_data['Close'].iloc[-1]
        
        # Get historical data for volatility calculation if not using custom
        if 'yes' not in use_custom_volatility:
            hist_data = stock.history(period="1y")
            if len(hist_data) > 30:  # Need enough data for volatility
                # Calculate historical volatility (annualized)
                returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized and in percentage
            else:
                volatility = 30  # Default if not enough data
        else:
            volatility = volatility_override
        
        # Convert inputs to proper format for Black-Scholes
        T = days_to_expiration / 365  # Time in years
        r = risk_free_rate / 100  # Convert percentage to decimal
        sigma = volatility / 100  # Convert percentage to decimal
        
        # Calculate option prices using Black-Scholes
        call_price = black_scholes(current_price, strike_price, T, r, sigma, "call")
        put_price = black_scholes(current_price, strike_price, T, r, sigma, "put")
        total_premium = call_price + put_price
        
        # Calculate breakeven points
        lower_breakeven, upper_breakeven = calculate_breakeven_points(strike_price, total_premium)
        
        # Calculate profit/loss at different stock prices
        price_range_min = max(0.1, current_price * 0.5)  # Avoid negative or zero prices
        price_range_max = current_price * 1.5
        price_range = np.linspace(price_range_min, price_range_max, 100)
        profit_df = calculate_straddle_profit(current_price, strike_price, call_price, put_price, price_range)
        
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
        
        # Add vertical line at strike price
        fig.add_shape(
            type="line",
            x0=strike_price,
            y0=min(profit_df['Profit/Loss']),
            x1=strike_price,
            y1=max(profit_df['Profit/Loss']),
            line=dict(color=colors['secondary'], width=2, dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            title=f"STRADDLE STRATEGY PROFIT/LOSS PROJECTION",
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
        
        # Prepare stock info display
        stock_info = html.Div([
            html.H4(f"{ticker} - {stock.info.get('shortName', ticker)}", style={'color': colors['accent'], 'marginTop': '0'}),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                html.Div([
                    html.P("CURRENT PRICE:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"${current_price:.2f}", style={'margin': '5px 0', 'fontSize': '18px'})
                ]),
                html.Div([
                    html.P("VOLATILITY:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"{volatility:.2f}%", style={'margin': '5px 0', 'fontSize': '18px'})
                ])
            ])
        ])
        
        # Prepare strategy results
        strategy_results = html.Div([
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5("CALL OPTION", style={'color': colors['accent'], 'marginTop': '0'}),
                    html.P(f"Price: ${call_price:.2f}", style={'margin': '5px 0'}),
                    html.P(f"Strike: ${strike_price:.2f}", style={'margin': '5px 0'}),
                ]),
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5("PUT OPTION", style={'color': colors['accent'], 'marginTop': '0'}),
                    html.P(f"Price: ${put_price:.2f}", style={'margin': '5px 0'}),
                    html.P(f"Strike: ${strike_price:.2f}", style={'margin': '5px 0'}),
                ]),
                html.Div(style={'minWidth': '150px', 'margin': '10px'}, children=[
                    html.H5("STRADDLE COST", style={'color': colors['accent'], 'marginTop': '0'}),
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
        
        return stock_info, strategy_results, fig
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            html.P(error_message, style={'color': colors['loss']}),
            html.P("Strategy analysis failed. Please check your inputs and try again.", style={'color': colors['loss']}),
            go.Figure()
        )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
