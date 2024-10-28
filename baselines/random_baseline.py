import argparse
import pandas as pd
import numpy as np
from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv
import os

load_dotenv()   

def load_data(file_path):
    return pd.read_json(file_path, orient='records', lines=True, dtype=False)

def main():
    parser = argparse.ArgumentParser(description="Generate random baseline predictions")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    args = parser.parse_args()

    dataset = args.dataset

    # Load the aggregated predictions dataset
    io_path = ConfigLoader.get_aggregated_predictions_dataset_path(dataset)
    predictions_df = load_data(io_path)

    # Generate random predictions based on the given distribution
    num_samples = len(predictions_df)
    random_predictions = np.random.choice([0, 1], size=num_samples)

    # Add the random predictions to the DataFrame
    predictions_df['baseline_random'] = random_predictions

    # Save the updated DataFrame
    predictions_df.to_json(io_path, orient='records', lines=True)

    print(f"Random baseline predictions saved to: {io_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Class distribution: {np.mean(random_predictions):.2f} (expected: 0.5)")

if __name__ == "__main__":
    main()