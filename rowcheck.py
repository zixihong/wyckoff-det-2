import csv

def get_row_lengths(file_path):
    row_lengths = []

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row_lengths.append(len(row))

    return row_lengths

# Example usage
file_path = 'pattern_data.csv'
lengths = get_row_lengths(file_path)
print("Row Lengths:", lengths)
