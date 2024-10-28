import os
import pandas as pd
from argparse import ArgumentParser
from src.ConfigLoader import ConfigLoader
import json
from dotenv import load_dotenv

load_dotenv()
# Constants

FOLDER_PATH = r'C:\Users\Daniel Neu\Documents\PC\Studium\Bachelorarbeit\dataset\model_outputs'  # Update this with your folder path

def count_null_values_in_jsonl(df, output_columns):
    # Dictionary to store null counts for each column
    null_counts = {col: df[col].isnull().sum() for col in output_columns}
    
    # Total number of rows
    total_count = len(df)
    
    return null_counts , total_count *  len(output_columns), sum(null_counts.values())


def get_models_in_group(model_group):
    print(model_group)
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]

def model_to_model_column(model : str):
    return ConfigLoader.get_model_config(model)["output_column"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--prompt_groups", type=str, help="The prompt groups to count the null values for.")
    parser.add_argument("--model_groups", type=str, help="The model groups to count the null values for.")
    parser.add_argument("--dataset", type=str, help="The dataset to count the null values for.")
    args = parser.parse_args()
    print(args)
    model_groups = args.model_groups.split(',')
    prompt_groups = args.prompt_groups.split(',')
    dataset = args.dataset

    models = [model for model_group in model_groups for model in get_models_in_group(model_group)]
    output_columns = [model_to_model_column(model) for model in models]
    prompts = [prompt for prompt_group in prompt_groups for prompt in get_prompts_in_group(prompt_group)]
    print(output_columns)
    print(prompts)

    total_sum = 0
    total_null_sum = 0
    result = {}
    prompt_result = {}

    for prompt in prompts:
        df = pd.read_json(ConfigLoader.get_prediction_dataset_path(dataset, prompt), orient='records', lines=True, dtype=False)
        null_counts, total_count, total_null_values = count_null_values_in_jsonl( df, output_columns)
        total_sum += total_count
        total_null_sum += total_null_values

        
        for column, count in null_counts.items():
            if column not in result:
                result[column] = 0
            result[column] += count

        prompt_result[prompt] = total_null_values


   
    print(total_sum)
    print(total_null_sum)
    print("-"*100)
    print("Results per Model:")
    print(result)
    print("-"*100)
    print("Results per Prompt:")
    print(prompt_result)