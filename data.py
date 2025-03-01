import csv
import random
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os

if os.path.exists("pattern_data.csv"):
    os.remove("pattern_data.csv")

def addFiller(original_list, output_file):
    """
    Add random filler points between consecutive points in the original list
    and pad with NaN values to ensure consistent length.
    """
    modified_data = []

    for i in range(len(original_list) - 1):
        start_point = original_list[i]
        end_point = original_list[i + 1]
        modified_data.append(start_point)
        
        for _ in range(random.randint(0, 5)):
            filler_value = random.uniform(min(start_point, end_point), max(start_point, end_point))
            modified_data.append(filler_value)

    # Add the last point from the original list
    modified_data.append(original_list[-1])

    # Pad with NaN values to ensure consistent length
    if len(modified_data) < 60:
        for i in range(60-len(modified_data)):
            modified_data.append(np.nan)

    # Save modified data to a CSV file
    with open(output_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(modified_data)

def generate_valid_pattern():
    """Generate a valid Wyckoff pattern"""
    pattern = []
    validity = True
    pattern.append(validity)
    
    # Initial price levels
    initial_low = random.uniform(0, 1)
    pattern.append(initial_low)
    initial_high = random.uniform(1, 2)
    pattern.append(initial_high)
    
    # Secondary test points
    sts = []
    for h in range(random.randint(1, 3)):
        st = random.uniform(0.9 * initial_low, 1.1 * initial_low)
        pattern.append(st)
        sts.append(st)

    # Spring/Sweep
    sweep = random.uniform(0, min(sts) * 0.8)
    pattern.append(sweep)

    # Sign of strength
    sos = random.uniform(initial_low, initial_high)
    pattern.append(sos)

    # Breakout
    bo = random.uniform(initial_high, 2)
    pattern.append(bo)
    
    return pattern

def generate_invalid_pattern():
    """Generate an invalid pattern that doesn't follow Wyckoff rules"""
    pattern = []
    validity = False
    pattern.append(validity)
    
    # Generate random points that don't follow the Wyckoff structure
    for k in range(random.randint(1, 9)):  # Fixed the missing argument
        pattern.append(random.uniform(0, 2))
    
    return pattern

# Generate valid patterns
print("Generating valid patterns...")
for j in range(50000):  # Reduced to 50k for balance
    pattern = generate_valid_pattern()
    addFiller(pattern, "pattern_data.csv")

# Generate invalid patterns
print("Generating invalid patterns...")
for i in range(50000):  # Reduced to 50k for balance
    pattern = generate_invalid_pattern()
    addFiller(pattern, "pattern_data.csv")

print("Processing and shuffling data...")
# Read the generated data
df_padded = pd.read_csv("pattern_data.csv", header=None)

# Handle any missing values
df_padded = df_padded.fillna(method='ffill').fillna(0)

# Load the padded DataFrame into a Dask DataFrame
ddf = dd.from_pandas(df_padded, npartitions=10)  # Increased partitions for better performance

# Shuffle the Dask DataFrame
ddf_shuffled = ddf.sample(frac=1, random_state=42)

# Save the shuffled Dask DataFrame to a new CSV file
ddf_shuffled.to_csv('pattern_data_padded_shuffled.csv', index=False, header=False, single_file=True)


