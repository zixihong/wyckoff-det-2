import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import argparse

def preprocess_data(data, pad_length=59):
    """Preprocess input data for prediction"""
    # Ensure data is the right length
    if len(data) < pad_length:
        # Pad with the last value
        padding = [data[-1]] * (pad_length - len(data))
        data = np.concatenate([data, padding])
    elif len(data) > pad_length:
        # Truncate
        data = data[:pad_length]
    
    # Reshape for the model
    return data.reshape(1, -1)

def predict_pattern(model, data, scaler):

    processed_data = preprocess_data(data)
    
    # Scale the data
    scaled_data = scaler.transform(processed_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0][0]
    
    return prediction

def plot_pattern(data, prediction, threshold=0.5):
    """Plot the pattern with prediction result"""
    is_valid = prediction >= threshold
    
    plt.figure(figsize=(12, 6))
    plt.plot(data, marker='o', linestyle='-', markersize=4)
    plt.title(f'Pattern Analysis - {"Valid" if is_valid else "Invalid"} Wyckoff Pattern ({prediction:.2f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Add confidence annotation
    confidence = prediction if is_valid else 1 - prediction
    plt.annotate(f'Confidence: {confidence:.2f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict if a price pattern follows Wyckoff principles')
    parser.add_argument('--file', type=str, help='CSV file containing price data')
    parser.add_argument('--column', type=int, default=0, help='Column index in CSV to use (default: 0)')
    parser.add_argument('--model', type=str, default='wyckoff_pattern_model.h5', help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Create a scaler (we'll fit it on the data for simplicity)
    scaler = StandardScaler()
    
    if args.file:
        # Load data from CSV
        try:
            df = pd.read_csv(args.file)
            data = df.iloc[:, args.column].values
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    else:
        # Use sample data for demonstration
        print("No input file provided. Using sample data...")
        # Sample Wyckoff accumulation pattern
        data = np.array([
            1.5, 1.2, 1.0, 0.9, 0.95, 0.85, 0.7, 0.9, 1.1, 1.3, 1.6, 1.8
        ])
    
    # Fit the scaler on this data (in real use, you'd load a pre-fit scaler)
    sample_data = np.random.rand(100, 59)
    scaler.fit(sample_data)
    
    # Make prediction
    prediction = predict_pattern(model, data, scaler)
    print(f"Prediction: {prediction:.4f}")
    print(f"This pattern is {'likely' if prediction >= 0.5 else 'unlikely'} to be a valid Wyckoff pattern.")
    
    # Plot the pattern
    plot_pattern(data, prediction)

if __name__ == "__main__":
    main() 