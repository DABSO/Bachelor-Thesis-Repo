from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv
import argparse
import pandas as pd
import os
AL_TYPE = "perplexity"


def remove_al_instances_from_training_set(type : str, dataset : str, prompt_group : str, model_group : str, id : str, eval_method : str, id_column : str):
    # Load the dataset
    dataset_config = ConfigLoader.get_dataset_config(dataset)
    relabeled_evaluation_dataset_path = ConfigLoader.build_relabeled_evaluation_dataset_path(dataset)
    relabeled_df = pd.read_json(relabeled_evaluation_dataset_path, orient='records', lines=True, dtype=False)

    # load the active learning dataset
    active_learning_datasets_folder = os.getenv("DATASET_DIR") + "/active_learning" + f"/{type}"
    # reecursively go through all files in the folder that are json files, load them, and concatenate them into one dataframe
    active_learning_df = pd.concat([pd.read_json(os.path.join(active_learning_datasets_folder, f), orient='records', dtype=False) for f in os.listdir(active_learning_datasets_folder) if f.endswith(".json")])
    active_learning_ids = active_learning_df[id_column].tolist()

    al_instances = relabeled_df[relabeled_df[id_column].isin(active_learning_ids)]

    # remove the instances from the training set
    filtered_relabeled_df = relabeled_df[~relabeled_df[id_column].isin(active_learning_ids)]
    # save the al instances to a seperate file
    filtered_relabeled_df.to_json(ConfigLoader.build_relabeled_evaluation_dataset_path(dataset), lines=True, orient='records')

    al_instances.to_json(ConfigLoader.build_relabeled_evaluation_dataset_path( type + "_" + dataset), lines=True, orient='records')


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, help="The type of active learning")
    parser.add_argument("--dataset", required=True, help="The dataset to remove the instances from")
    parser.add_argument("--prompt_group", required=True, help="The prompt group")
    parser.add_argument("--model_group", required=True, help="The model group")
    parser.add_argument("--id", required=True, help="The id of the model")
    parser.add_argument("--eval_method", choices=['bool', 'precision_recall'], default='bool', help="The evaluation method")
    args = parser.parse_args()
    id_column = ConfigLoader.get_dataset_config(args.dataset)["id_column"]
    remove_al_instances_from_training_set(args.type, args.dataset, args.prompt_group, args.model_group, args.id, args.eval_method, id_column)