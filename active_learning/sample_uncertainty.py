from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv
import pandas as pd
import argparse
from joblib import load
import json
import os

load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--eval_method", choices=['bool', 'precision_recall'], default='bool', help="Evaluation method")
parser.add_argument("--n", type=int, default=100, help="Number of instances to get")
parser.add_argument("--classifier", choices=['mlp','random_forest'],required=True, help="Classifier to use")
parser.add_argument("--model_group", required=True, help="Name of the model group")
parser.add_argument("--prompt_group", required=True, help="Name of the prompt group")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument("--id", required=True, help="Identifier for the model")
args = parser.parse_args()
dataset_name = "human_annotated_train"
dataset_config = ConfigLoader.get_dataset_config(dataset_name)
train_dataset_path = ConfigLoader.get_aggregated_predictions_dataset_path(dataset_name)
classifier_name = ConfigLoader.get_classifier_name(args.classifier,"" ,[args.model_group, args.prompt_group, args.eval_method, args.id])
relabeled_dataset_path = ConfigLoader.get_relabeled_evaluation_dataset_path(dataset_name)
relabeled_dataset = pd.read_json(relabeled_dataset_path, orient="records", lines=True, dtype=False)
id_column = dataset_config["id_column"]
relabeled_ids = relabeled_dataset[id_column].tolist()




train_dataset = pd.read_json(train_dataset_path, orient="records", lines=True, dtype=False)

assert classifier_name in train_dataset.columns, f"Classifier {classifier_name} not found in train dataset"

# Calculate the absolute distance from 0.5 for each prediction
train_dataset['distance_from_0.5'] = abs(train_dataset[classifier_name] - 0.5)

# Sort the dataset by the distance (ascending order)
sorted_dataset = train_dataset.sort_values('distance_from_0.5')


# Filter out instances that are already in relabeled_ids
new_instances = sorted_dataset[~sorted_dataset[id_column].isin(relabeled_ids)]
# Get the top n instancess
top_n_instances = new_instances.head(args.n)

# Create a list of dictionaries containing the required information
results = []
for _, row in top_n_instances.iterrows():
    results.append({
        "id": row[id_column],
        "score": round(row[classifier_name], 4),
        "distance_from_0.5": round(row['distance_from_0.5'], 4)
    })

# Create the directory if it doesn't exist
os.makedirs(os.getenv("DATASET_DIR") + "/active_learning/uncertainty", exist_ok=True)

# Save the results to a JSON file
output_file = f"{os.getenv("DATASET_DIR")}/active_learning/uncertainty/{args.classifier}_{args.model_group}_{args.prompt_group}_{args.eval_method}_{args.id}_top_{args.n}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")

# Print the results
print(f"Top {args.n} instances with highest uncertainty(closest to 0.5) that are not yet relabeled:")
for item in results:
    print(f"ID: {item['id_column']}, Score: {item['score']:.4f}, Distance from 0.5: {item['distance_from_0.5']:.4f}")

