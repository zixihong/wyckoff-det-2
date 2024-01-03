import csv
import random
import pandas as pd
import numpy as np
import dask.dataframe as dd

def addFiller(original_list, output_file):
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

    if (len(modified_data) < 50):
        for i in range(50-len(modified_data)):
            modified_data.append(np.nan)

    # Save modified data to a CSV file
    with open(output_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(modified_data)

for j in range(100):
    pattern = []
    validity = True
    pattern.append(validity)
    initial_low = random.uniform(0, 1)
    pattern.append(initial_low)
    initial_high = random.uniform(1, 2)
    pattern.append(initial_high)
    sts = []
    for h in range(random.randint(1, 3)):
        st = random.uniform(0.9 * initial_low, 1.1 * initial_low)
        pattern.append(st)
        sts.append(st)

    sweep = random.uniform(0, min(sts) * 0.8)
    pattern.append(sweep)

    sos = random.uniform(initial_low, initial_high)
    pattern.append(sos)

    bo = random.uniform(initial_high, 2)
    pattern.append(bo)

    addFiller(pattern, "pattern_data.csv")

for i in range(100):
    pattern = []
    validity = False
    pattern.append(validity)
    for k in range(random.randint(7, 9)):
        pattern.append(random.uniform(0, 2))

    addFiller(pattern, "pattern_data.csv")


df_padded = pd.read_csv("pattern_data.csv")

# Load the padded DataFrame into a Dask DataFrame
ddf = dd.from_pandas(df_padded, npartitions=1)

# Shuffle the Dask DataFrame
ddf_shuffled = ddf.sample(frac=1)

# Save the shuffled Dask DataFrame to a new CSV file
ddf_shuffled.to_csv('pattern_data_padded_shuffled.csv', index=False, header=False, single_file=True)
