import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download Bitcoin data from Yahoo Finance
data = yf.download("BTC-USD", start="2018-01-01", end="2024-10-01")
data['return'] = data['Adj Close'].pct_change()

# Define 2 states based on return direction
data['state'] = np.where(data['return'] >= 0, 'up', 'down')

# Shift states to calculate second-order transitions
data['prev_state'] = data['state'].shift(1)
data['two_prev_state'] = data['state'].shift(2)

# Remove rows with NaN values caused by shifting
data.dropna(inplace=True)

# Define states list
states = ['up', 'down']

# Define all possible second-order transitions
second_order_transitions = [(s1, s2, s3) for s1 in states for s2 in states for s3 in states]

# Calculate second-order transition probabilities
transition_counts = {}
for (s1, s2, s3) in second_order_transitions:
    count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2) & (data['state'] == s3)])
    total_count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2)])
    probability = count / total_count if total_count > 0 else 0
    transition_counts[(s1, s2, s3)] = probability

# Create a DataFrame for the second-order transition matrix
transition_matrix = pd.DataFrame(
    {
        (s1, s2): [transition_counts[(s1, s2, s3)] for s3 in states]
        for s1 in states for s2 in states
    },
    index=states
)

print("Second-Order Transition Matrix for Bitcoin States:")
print(transition_matrix)

# Visualize the transition matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(transition_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Second-Order Transition Probability Matrix for Bitcoin States")
plt.xlabel("(Previous State 1, Previous State 2)")
plt.ylabel("Current State")
plt.show()

# Trading signal generator for 2 states
def generate_trading_signal(row, threshold=0.5):
    """Generate contrarian trading signals based on transition probabilities."""
    prev1, prev2 = row['prev_state'], row['two_prev_state']
    if (prev1, prev2) in transition_matrix.columns:
        # Extract transition probabilities for the current previous states
        prob_up = transition_matrix[(prev1, prev2)]['up']
        prob_down = transition_matrix[(prev1, prev2)]['down']
        
        # Contrarian trading strategy
        if prob_up < threshold:
            return "Buy"  # Low probability of uptrend, so contrarian Buy
        elif prob_down < threshold:
            return "Sell"  # Low probability of downtrend, so contrarian Sell
        else:
            return "Hold"
    else:
        return "Hold"

# Apply the trading signal function
data['signal'] = data.apply(generate_trading_signal, axis=1)

# Calculate daily returns based on signals
data['strategy_return'] = np.where(data['signal'] == 'Buy', data['return'],
                                 np.where(data['signal'] == 'Sell', -data['return'], 0))

# Calculate cumulative returns
data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod()
data['cumulative_btc_return'] = (1 + data['return']).cumprod()

# Calculate Maximum Drawdown
def calculate_max_drawdown(returns):
    """Calculate the maximum drawdown from a series of returns."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    return drawdowns.min()

# Calculate Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """Calculate the annualized Sharpe Ratio."""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

# Calculate performance metrics
strategy_max_drawdown = calculate_max_drawdown(data['strategy_return'])
btc_max_drawdown = calculate_max_drawdown(data['return'])

strategy_sharpe = calculate_sharpe_ratio(data['strategy_return'])
btc_sharpe = calculate_sharpe_ratio(data['return'])

# Final cumulative return values
final_strategy_return = data['cumulative_strategy_return'].iloc[-1]
final_btc_return = data['cumulative_btc_return'].iloc[-1]

# Print performance metrics
print("\nPerformance Metrics:")
print("-" * 50)
print("Strategy Performance:")
print(f"Cumulative Return: {(final_strategy_return - 1):.2%}")
print(f"Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Maximum Drawdown: {strategy_max_drawdown:.2%}")
print("\nBuy & Hold Performance:")
print(f"Cumulative Return: {(final_btc_return - 1):.2%}")
print(f"Sharpe Ratio: {btc_sharpe:.2f}")
print(f"Maximum Drawdown: {btc_max_drawdown:.2%}")

# Display a portion of the data with trading signals and returns
print("\nLast 20 days of trading signals and returns:")
print(data[['return', 'state', 'prev_state', 'two_prev_state', 'signal', 
            'strategy_return', 'cumulative_strategy_return', 'cumulative_btc_return']].tail(20))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['cumulative_strategy_return'], label='Strategy Cumulative Return', color='blue')
plt.plot(data.index, data['cumulative_btc_return'], label='Bitcoin Buy & Hold Return', color='orange')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of Contrarian Strategy vs. Bitcoin Buy & Hold")
plt.legend()
plt.show()

# Plot drawdown over time
plt.figure(figsize=(12, 6))
strategy_drawdown = data['cumulative_strategy_return'] / data['cumulative_strategy_return'].expanding().max() - 1
btc_drawdown = data['cumulative_btc_return'] / data['cumulative_btc_return'].expanding().max() - 1

plt.plot(data.index, strategy_drawdown, label='Strategy Drawdown', color='blue')
plt.plot(data.index, btc_drawdown, label='Bitcoin Drawdown', color='orange')
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.title("Drawdown Over Time")
plt.legend()
plt.grid(True)
plt.show()