from src.ConfigLoader import ConfigLoader
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
dataset = parser.parse_args().dataset
id_column = ConfigLoader.get_dataset_config(dataset)["id_column"]
#dataset_path 
path = ConfigLoader.build_model_output_dataset_path(dataset, prompt_name="baseline")

model_df = pd.read_json(path, dtype=False, orient="records", lines=True)

dataset_config = ConfigLoader.get_dataset_config(dataset)

aggregated_path = ConfigLoader.build_aggregated_predictions_dataset_path(dataset)

aggregated_df = pd.read_json(aggregated_path, dtype=False, lines=True, orient="records")

# Ensure IDs are aligned
assert (model_df[id_column] == aggregated_df[id_column]).all(), "IDs are not aligned"

for col in model_df.columns:
    if col == id_column:
        continue
    else:
        # for values that are none specify the max - average value to create a better order 
        aggregated_df["baseline_"+col] = model_df[col]
        average = aggregated_df["baseline_"+col].mean()
        print(average)
        max = aggregated_df["baseline_"+col].max()
        print(max)
        aggregated_df["baseline_"+col] = aggregated_df["baseline_"+col].map(lambda v: max- v if v is not None and not pd.isna(v) else max-average)
        
        print(aggregated_df["baseline_"+col].head(10))



aggregated_df.to_json(aggregated_path, lines=True, orient="records")