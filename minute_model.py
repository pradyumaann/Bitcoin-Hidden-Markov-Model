import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download minute-level Bitcoin data (yfinance limits to 7 days of minute data)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
data = yf.download("BTC-USD", start=start_date, end=end_date, interval='1m')
data['return'] = data['Adj Close'].pct_change()

# Define multiple states based on return magnitudes
def classify_return(ret):
    if ret <= -0.005:  # Strong downward movement (-0.5% or more)
        return 'strong_down'
    elif -0.005 < ret <= -0.001:  # Moderate downward movement
        return 'moderate_down'
    elif -0.001 < ret < 0:  # Slight downward movement
        return 'slight_down'
    elif ret == 0:  # No movement
        return 'neutral'
    elif 0 < ret < 0.001:  # Slight upward movement
        return 'slight_up'
    elif 0.001 <= ret < 0.005:  # Moderate upward movement
        return 'moderate_up'
    else:  # Strong upward movement (0.5% or more)
        return 'strong_up'

# Apply state classification
data['state'] = data['return'].apply(classify_return)

# Shift states to calculate second-order transitions
data['prev_state'] = data['state'].shift(1)
data['two_prev_state'] = data['state'].shift(2)

# Remove rows with NaN values caused by shifting
data.dropna(inplace=True)

# Define states list
states = ['strong_down', 'moderate_down', 'slight_down', 'neutral', 
          'slight_up', 'moderate_up', 'strong_up']

# Define all possible second-order transitions
second_order_transitions = [(s1, s2, s3) for s1 in states for s2 in states for s3 in states]

# Calculate second-order transition probabilities
transition_counts = {}
for (s1, s2, s3) in second_order_transitions:
    count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2) & (data['state'] == s3)])
    total_count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2)])
    probability = count / total_count if total_count > 0 else 0
    transition_counts[(s1, s2, s3)] = probability

# Create a simplified transition matrix for visualization
# We'll use the most recent state and current state for better visibility
simplified_matrix = pd.pivot_table(
    data,
    values='return',
    index='state',
    columns='prev_state',
    aggfunc=lambda x: len(x) / len(data)
).fillna(0)

# Visualize the simplified transition matrix as a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(simplified_matrix, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("First-Order Transition Probability Matrix for Bitcoin States (Minute Level)")
plt.xlabel("Previous State")
plt.ylabel("Current State")
plt.tight_layout()
plt.show()

# Modified trading signal generator for multiple states
def generate_trading_signal(row):
    """Generate trading signals based on state transitions."""
    current_state = row['state']
    prev_state = row['prev_state']
    
    # Strong signals for trend reversals
    if prev_state in ['strong_down', 'moderate_down'] and current_state in ['slight_up', 'moderate_up', 'strong_up']:
        return "Buy"  # Potential trend reversal from downward to upward
    elif prev_state in ['strong_up', 'moderate_up'] and current_state in ['slight_down', 'moderate_down', 'strong_down']:
        return "Sell"  # Potential trend reversal from upward to downward
    elif current_state == 'strong_down':
        return "Buy"  # Contrarian buy on strong downward movement
    elif current_state == 'strong_up':
        return "Sell"  # Contrarian sell on strong upward movement
    else:
        return "Hold"

# Apply the trading signal function
data['signal'] = data.apply(generate_trading_signal, axis=1)

# Calculate minute returns based on signals
data['strategy_return'] = np.where(data['signal'] == 'Buy', data['return'],
                                 np.where(data['signal'] == 'Sell', -data['return'], 0))

# Calculate cumulative returns
data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod()
data['cumulative_btc_return'] = (1 + data['return']).cumprod()

# Calculate state distribution
state_distribution = data['state'].value_counts() / len(data)
print("\nState Distribution:")
print(state_distribution)

# Calculate transition frequencies
print("\nMost Common State Transitions:")
transition_counts = data.groupby(['prev_state', 'state']).size().sort_values(ascending=False).head(10)
print(transition_counts)

# Calculate performance metrics
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    return drawdowns.min()

def calculate_minute_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return np.sqrt(525600) * (excess_returns.mean() / excess_returns.std())

# Calculate metrics
strategy_max_drawdown = calculate_max_drawdown(data['strategy_return'])
btc_max_drawdown = calculate_max_drawdown(data['return'])
strategy_sharpe = calculate_minute_sharpe_ratio(data['strategy_return'])
btc_sharpe = calculate_minute_sharpe_ratio(data['return'])
final_strategy_return = data['cumulative_strategy_return'].iloc[-1]
final_btc_return = data['cumulative_btc_return'].iloc[-1]

# Print performance metrics
print("\nPerformance Metrics (Last 7 Days, Minute-Level):")
print("-" * 50)
print("Strategy Performance:")
print(f"Cumulative Return: {(final_strategy_return - 1):.2%}")
print(f"Minute-Adjusted Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Maximum Drawdown: {strategy_max_drawdown:.2%}")
print("\nBuy & Hold Performance:")
print(f"Cumulative Return: {(final_btc_return - 1):.2%}")
print(f"Minute-Adjusted Sharpe Ratio: {btc_sharpe:.2f}")
print(f"Maximum Drawdown: {btc_max_drawdown:.2%}")

# Calculate signal distribution
signal_distribution = data['signal'].value_counts()
print("\nTrading Signal Distribution:")
print(signal_distribution)

# Plot cumulative returns
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['cumulative_strategy_return'], label='Strategy Cumulative Return', color='blue')
plt.plot(data.index, data['cumulative_btc_return'], label='Bitcoin Buy & Hold Return', color='orange')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of Multi-State Strategy vs. Bitcoin Buy & Hold (Minute-Level)")
plt.legend()
plt.grid(True)
plt.show()

# Plot state distribution
plt.figure(figsize=(12, 6))
state_distribution.plot(kind='bar')
plt.title("Distribution of Bitcoin Price States")
plt.xlabel("State")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create state transition flow visualization using Sankey diagram
from matplotlib.sankey import Sankey

# Get transition counts for visualization
transitions = data.groupby(['prev_state', 'state']).size().reset_index()
transitions.columns = ['source', 'target', 'value']
transitions['value'] = transitions['value'] / len(data)

# Plot Sankey diagram for top transitions
plt.figure(figsize=(15, 10))
sankey = Sankey(ax=plt.gca(), scale=0.01, offset=0.1)
sankey.add(flows=transitions['value'][:10],
           labels=transitions['source'][:10],
           orientations=[0]*10)
plt.title("Top State Transitions Flow")
plt.axis('off')
plt.show()