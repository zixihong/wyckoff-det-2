import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import argparse
import os
from datetime import datetime

class WyckoffPatternTrader:
    def __init__(self, model_path, window_size=60, threshold=0.7):
        """Initialize the Wyckoff pattern trader"""
        self.model = load_model(model_path)
        self.window_size = window_size
        self.threshold = threshold
        self.scaler = StandardScaler()
        
        # Create a dummy dataset to fit the scaler
        dummy_data = np.random.rand(100, window_size-1)
        self.scaler.fit(dummy_data)
        
        # Initialize price history
        self.price_history = []
        
        # Trading state
        self.in_position = False
        self.entry_price = 0
        self.trade_history = []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def preprocess_data(self, data):
        """Preprocess the price data for prediction"""
        if len(data) < self.window_size - 1:
            # Not enough data points yet
            return None
        
        # Get the most recent window
        window = data[-(self.window_size-1):]
        
        # Scale the data
        scaled_window = self.scaler.transform(window.reshape(1, -1))
        
        return scaled_window
    
    def detect_pattern(self, price):
        """Detect if the current price window forms a Wyckoff pattern"""
        # Add the new price to history
        self.price_history.append(price)
        
        # Check if we have enough data
        if len(self.price_history) < self.window_size:
            return False, 0
        
        # Preprocess the data
        data = np.array(self.price_history[-(self.window_size-1):])
        processed_data = self.preprocess_data(data)
        
        if processed_data is None:
            return False, 0
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0][0]
        
        # Check if it's a valid pattern with high confidence
        is_valid = prediction >= self.threshold
        
        return is_valid, prediction
    
    def execute_trade(self, price, timestamp):
        """Execute a trade based on pattern detection"""
        is_valid_pattern, confidence = self.detect_pattern(price)
        
        # Trading logic
        if is_valid_pattern and not self.in_position:
            # Enter position
            self.in_position = True
            self.entry_price = price
            trade = {
                'action': 'BUY',
                'price': price,
                'timestamp': timestamp,
                'confidence': confidence
            }
            self.trade_history.append(trade)
            print(f"[{timestamp}] BUY signal at {price:.2f} (confidence: {confidence:.2f})")
            return "BUY", confidence
        
        elif self.in_position:
            # Simple exit strategy: exit if price increases by 5% or decreases by 2%
            profit_pct = (price - self.entry_price) / self.entry_price * 100
            
            if profit_pct >= 5 or profit_pct <= -2:
                self.in_position = False
                trade = {
                    'action': 'SELL',
                    'price': price,
                    'timestamp': timestamp,
                    'profit_pct': profit_pct
                }
                self.trade_history.append(trade)
                print(f"[{timestamp}] SELL signal at {price:.2f} (profit: {profit_pct:.2f}%)")
                return "SELL", profit_pct
        
        return "HOLD", 0
    
    def plot_trades(self):
        """Plot the price history with trade signals"""
        if not self.price_history:
            return
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.price_history, label='Price')
        
        # Plot buy signals
        buy_indices = []
        buy_prices = []
        
        # Plot sell signals
        sell_indices = []
        sell_prices = []
        
        # Extract trade information
        for i, trade in enumerate(self.trade_history):
            if trade['action'] == 'BUY':
                # Find the index in price_history
                try:
                    idx = self.price_history.index(trade['price'])
                    buy_indices.append(idx)
                    buy_prices.append(trade['price'])
                except ValueError:
                    # Price might not be exactly in the list due to floating point
                    pass
            elif trade['action'] == 'SELL':
                try:
                    idx = self.price_history.index(trade['price'])
                    sell_indices.append(idx)
                    sell_prices.append(trade['price'])
                except ValueError:
                    pass
        
        # Plot the signals
        plt.scatter(buy_indices, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
        plt.scatter(sell_indices, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
        
        plt.title('Wyckoff Pattern Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/trading_signals_{timestamp}.png')
        
        # Save trade history to CSV
        if self.trade_history:
            pd.DataFrame(self.trade_history).to_csv(f'results/trade_history_{timestamp}.csv', index=False)

def simulate_trading(trader, data_file=None, n_steps=1000):
    """Simulate trading with historical or generated data"""
    if data_file:
        # Use historical data
        try:
            df = pd.read_csv(data_file)
            prices = df.iloc[:, 0].values
            n_steps = min(n_steps, len(prices))
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    else:
        # Generate random price data with some trend
        prices = [100]
        for _ in range(n_steps-1):
            # Random walk with slight upward bias
            change = np.random.normal(0.05, 1.0)
            new_price = max(0.1, prices[-1] + change)
            prices.append(new_price)
    
    # Simulate trading
    print(f"Starting trading simulation with {n_steps} price points...")
    
    for i in range(n_steps):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price = prices[i]
        
        # Execute trade
        action, info = trader.execute_trade(price, timestamp)
        
        # Add some delay for real-time simulation feel
        time.sleep(0.01)
        
        # Print progress
        if i % 100 == 0:
            print(f"Processed {i}/{n_steps} price points...")
    
    # Plot the results
    trader.plot_trades()
    
    # Calculate performance metrics
    total_trades = len([t for t in trader.trade_history if t['action'] == 'SELL'])
    profitable_trades = len([t for t in trader.trade_history if t['action'] == 'SELL' and t.get('profit_pct', 0) > 0])
    
    if total_trades > 0:
        win_rate = profitable_trades / total_trades * 100
        print(f"\nTrading Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        # Calculate total return
        total_return = sum([t.get('profit_pct', 0) for t in trader.trade_history if t['action'] == 'SELL'])
        print(f"Total Return: {total_return:.2f}%")
    else:
        print("No completed trades in this simulation.")

def main():
    parser = argparse.ArgumentParser(description='Wyckoff Pattern Trading Simulator')
    parser.add_argument('--model', type=str, default='wyckoff_pattern_model.h5', help='Path to the trained model')
    parser.add_argument('--data', type=str, help='CSV file containing price data (optional)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of price steps to simulate')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for pattern detection')
    
    args = parser.parse_args()
    
    # Initialize the trader
    trader = WyckoffPatternTrader(args.model, threshold=args.threshold)
    
    # Run simulation
    simulate_trading(trader, args.data, args.steps)

if __name__ == "__main__":
    main() 