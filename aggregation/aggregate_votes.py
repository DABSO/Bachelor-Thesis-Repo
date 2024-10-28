import json
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

from src.ArgParsers.EvaluationArgParser import EvaluationArgParser
from src.ConfigLoader import ConfigLoader
import argparse
load_dotenv()

# Constants
ALL_KEYWORD = "[all]"
VOTE_COLUMNS = ["majority_vote", "unanimous_vote"]

def load_data(file_path: str) -> pd.DataFrame:
    try:
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def get_prediction_columns(df: pd.DataFrame, models: List[str]) -> List[str]:
    
    return [
        config["output_column"] for model_name in models
        if (config := ConfigLoader.get_model_config(model_name))["output_column"] in df.columns
    ]

def evaluate_predictions(df: pd.DataFrame, prediction_columns: List[str], eval_func: callable, parsing_func: callable, label_col: str, keep_columns : List[str]) -> pd.DataFrame:
    prediction_df = df[keep_columns].copy()

    for column in prediction_columns:
        prediction_df[column] = df.apply(
            lambda row: eval_func(parsing_func(row[column]), row[label_col]) if row[column] is not None else None,
            axis=1
        )
    
    # get row with id 19_2
    print("19_2",prediction_df[prediction_df["id"] == "19_2"])
    
    return prediction_df

def calculate_votes(prediction_df: pd.DataFrame, prediction_columns: List[str], prompt_group, model_group, invalidate_nan) -> pd.DataFrame:
    
    def majority_vote(row):
        votes = [0.5 if row[col] is None else int(row[col]) for col in prediction_columns]
        if invalidate_nan:
            if any(vote == 0.5 for vote in votes):
                return None
        if not votes:
            return None
        return sum(votes) > len(votes) / 2

    def unanimous_vote(row):
        votes = [0.5 if row[col] is None else int(row[col]) for col in prediction_columns]
        if invalidate_nan:
            if any(vote == 0.5 for vote in votes):
                return None
        if all(vote == 1 for vote in votes):
            return True
        elif all(vote == 0 for vote in votes):
            return False
        
        return None

    def percentage_vote(row):
        
           
        votes = [0.5 if row[col] is None else int(row[col]) for col in prediction_columns]
        if invalidate_nan:
            if any(vote == 0.5 for vote in votes):
                return None	
        return sum(votes) / len(votes) if votes else None

    
        


    prediction_df['majority_vote' + "_"+prompt_group+ "-"+ model_group] = prediction_df.apply(majority_vote, axis=1)
    prediction_df['unanimous_vote'+ "_"+prompt_group+ "-"+ model_group] = prediction_df.apply(unanimous_vote, axis=1)
    prediction_df['percentage_vote'+ "_"+prompt_group+ "-"+ model_group] = prediction_df.apply(percentage_vote, axis=1)

    return prediction_df

def merge_prompt_evaluations(eval_dfs: Dict[str, pd.DataFrame], keep_columns : List[str], prompt_group, model_group, merged_df = None,invalidate_nan=False) -> pd.DataFrame:
    if not eval_dfs:
        return pd.DataFrame()
    if merged_df is None:
        merged_df = next(iter(eval_dfs.values()))[keep_columns].copy()

    for i in range(len(merged_df)):
        rows = {prompt: df.iloc[i] for prompt, df in eval_dfs.items()}  

        majority_votes = [row["majority_vote"+ "_"+prompt_group+ "-"+ model_group ] for row in rows.values() if row["majority_vote" + "_"+prompt_group+ "-"+ model_group] is not None and not pd.isna(row["majority_vote" + "_"+prompt_group+ "-"+ model_group])]
        majority_votes_with_nan = [row["majority_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values()]
        if invalidate_nan and any(vote is None or pd.isna(vote)  for vote in majority_votes_with_nan):
            merged_df.at[i, "majority_vote" + "_"+prompt_group+ "-"+ model_group + (f"_no_nan" if invalidate_nan else "")] = None
        else:
            merged_df.at[i, "majority_vote" + "_" +prompt_group+ "-" + model_group + (f"_no_nan" if invalidate_nan else "")] = (sum(majority_votes) > len(majority_votes) / 2) if majority_votes else None

        unanimous_votes = [row["unanimous_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values() if row["percentage_vote"+ "_"+prompt_group+ "-"+ model_group] ]
        unanimous_votes_with_nan = [row["unanimous_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values()]
        if invalidate_nan and any(vote is None or pd.isna(vote)  for vote in unanimous_votes_with_nan):
            merged_df.at[i, "unanimous_vote"+ "_"+prompt_group+ "-"+ model_group + (f"_no_nan" if invalidate_nan else "")] = None
        else:
            merged_df.at[i, "unanimous_vote"+ "_" +prompt_group+ "-" + model_group + (f"_no_nan" if invalidate_nan else "")] = True if all(unanimous_votes) else False if all(vote is False for vote in unanimous_votes) else None

        
        percentage_votes = [row["percentage_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values() if row["percentage_vote"+ "_"+prompt_group+ "-"+ model_group] is not None and not pd.isna(row["percentage_vote"+ "_"+prompt_group+ "-"+ model_group])]
        percentage_votes_with_nan = [row["percentage_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values()]
        if invalidate_nan and any(vote is None or pd.isna(vote) for vote in percentage_votes_with_nan):
            merged_df.at[i, "percentage_vote"+ "_"+prompt_group+ "-"+ model_group +( f"_no_nan" if invalidate_nan else "")] = None
        else:
            merged_df.at[i, "percentage_vote"+ "_" +prompt_group+ "-" + model_group + (f"_no_nan" if invalidate_nan else "")] = sum(percentage_votes) / len(percentage_votes) if percentage_votes else None
        
     
        model_majority_votes = [row["majority_vote" + "_"+prompt_group+ "-"+ model_group] for row in rows.values() if row["majority_vote"+ "_"+prompt_group+ "-"+ model_group] is not None]
        model_majority_votes_with_nan = [row["majority_vote"+ "_"+prompt_group+ "-"+ model_group] for row in rows.values()]
        if invalidate_nan and any(vote is None or pd.isna(vote) for vote in model_majority_votes_with_nan):
            merged_df.at[i, "model_majority_prompt_unanimous_vote"+ "_" +prompt_group+ "-" + model_group + (f"_no_nan" if invalidate_nan else "")] = None
        else:
            merged_df.at[i, "model_majority_prompt_unanimous_vote"+ "_" +prompt_group+ "-" + model_group + (f"_no_nan" if invalidate_nan else "")] = all(model_majority_votes) if model_majority_votes else None
        

    return merged_df

def save_predictions(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_json(path, orient='records', lines=True)
        print(f"Saved predictions to: {path}")
    except Exception as e:
        print(f"Error saving predictions to {path}: {e}")

def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--model_group", required=True, help="Name of the model group")
    parser.add_argument("--prompt_group", required=True, help="Name of the prompt group")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--invalidate_nan", action="store_true", help="Invalidate nan values")
    args = parser.parse_args()

    prompts = get_prompts_in_group(args.prompt_group)
    if len(prompts) == 0:
        raise Exception("No prompts in this group")
    print("prompts: ", prompts)
    models = get_models_in_group(args.model_group)
    if len(models) == 0:
        raise Exception("No models in this group")
    datasets = ConfigLoader.load_available_datasets()

    for dataset in datasets:
        
        if args.dataset != ALL_KEYWORD and args.dataset != dataset:
            continue

        dataset_config = ConfigLoader.get_dataset_config(dataset)
        keep_columns = dataset_config["keep_columns"]
        label_col = dataset_config["label_column"]

        eval_dfs = {}

        for prompt in prompts:
            
            dataset_path = ConfigLoader.build_model_output_dataset_path(dataset, prompt)
            df = load_data(dataset_path)
            if df is None:
                raise ValueError("df ", dataset_path, "does not exist")


            prediction_columns = get_prediction_columns(df, models)
            if len(prediction_columns) != len(models):
                for model in models:
                    if out_col := ConfigLoader.get_model_config(model)["output_column"] not in df.columns:
                        raise ValueError(f"output_column {out_col} missing in df", dataset_path)

            prompt_config = ConfigLoader.get_prompt_config(prompt)
            eval_func, parsing_func = prompt_config["evaluation_function"], prompt_config["parsing_function"]
            print("DF columns", df.columns) 
            print("Prediction columns", prediction_columns)
            prediction_df = evaluate_predictions(df, prediction_columns, eval_func, parsing_func, label_col, keep_columns)
            prediction_df = calculate_votes(prediction_df, prediction_columns, args.prompt_group, args.model_group, args.invalidate_nan)

            eval_dfs[prompt] = prediction_df

            prediction_path = ConfigLoader.build_prediction_dataset_path(dataset, prompt)
            save_predictions(prediction_df, prediction_path)

        merged_df = None
        multiprompt_prediction_path = ConfigLoader.build_aggregated_predictions_dataset_path(dataset)
        try:
            merged_df = pd.read_json(multiprompt_prediction_path, lines=True, orient="records", dtype=False)
        except Exception:
            merged_df = None
            

        multiprompt_prediction_df = merge_prompt_evaluations(eval_dfs, keep_columns, args.prompt_group, args.model_group, merged_df=merged_df, invalidate_nan=args.invalidate_nan)

       
        
        save_predictions(multiprompt_prediction_df, multiprompt_prediction_path)

if __name__ == "__main__":
    main()