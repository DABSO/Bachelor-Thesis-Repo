from argparse import ArgumentParser
import pandas as pd
from dotenv import load_dotenv
from src.ConfigLoader import ConfigLoader
import os

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, help="The dataset to count the errors for.")

args = parser.parse_args()
dataset = args.dataset
if dataset == "all":
    dataset_folder= os.getenv("DATASET_DIR") + "/evaluations/relabeled"
    dataset_files = [os.path.splitext(f)[0] for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]
    dataset_files = [os.path.basename(f) for f in dataset_files]
    print(dataset_files)
else:
    dataset_files = [dataset]

total_matches = 0
total_rows = 0

for dataset_file in dataset_files:
    df = pd.read_json(ConfigLoader.get_relabeled_evaluation_dataset_path(dataset_file), orient='records', lines=True, dtype=False)
    
    matches_count = df['matches_original'].sum()
    total_count = len(df)
    percentage = (matches_count / total_count) * 100
    
    print(f"\nDataset: {dataset_file}")
    print(f"Matches: {matches_count}")
    print(f"Total rows: {total_count}")
    print(f"Percentage of matches: {percentage:.2f}%")
    
    total_matches += matches_count
    total_rows += total_count

if len(dataset_files) > 1:
    total_percentage = (total_matches / total_rows) * 100
    print(f"\nOverall statistics:")
    print(f"Total matches: {total_matches}")
    print(f"Total rows: {total_rows}")
    print(f"Overall percentage of matches: {total_percentage:.2f}%")

if __name__ == "__main__":
    print(args)



