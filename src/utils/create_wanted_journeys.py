import csv
import random
from itertools import combinations

def create_wanted_journeys(input_file, output_file, num_pairs):
    # Read the 4th column from the input CSV file
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip header row
        data = [row[3] for row in reader if len(row) > 3]

    # Generate all unique unordered pairs
    all_pairs = list(combinations(data, 2))

    # Limit the number of pairs if needed.
    if num_pairs > len(all_pairs):
        num_pairs = len(all_pairs)

    # Randomly sample the desired number of pairs
    sampled_pairs = random.sample(all_pairs, num_pairs)

    # Write the pairs to the output CSV file
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        for pair in sampled_pairs:
            writer.writerow(pair)


# Example usage
input_file = 'data/airports.csv'
output_file = 'data/wanted_journeys3.csv'
num_pairs = int(74**1.5)
create_wanted_journeys(input_file, output_file, num_pairs)
