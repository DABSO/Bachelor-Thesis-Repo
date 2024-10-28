from src.ConfigLoader import ConfigLoader
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
class Evaluator:
    ALL_KEYWORD = "[all]"
    VOTE_COLUMNS = ["majority_vote", "unanimous_vote"]
        
    @staticmethod
    def evaluate( models, dataset, prompts):
        
        dataset_config = ConfigLoader.get_dataset_config(dataset)
        keep_columns = dataset_config["keep_columns"]
        label_col = dataset_config["label_column"]

        eval_dfs = {}
        no_predictions = True

        for prompt in prompts:
            

            dataset_path = ConfigLoader.build_model_output_dataset_path(dataset, prompt)
            df = Evaluator.load_data(dataset_path)
            if df is None:
                continue

            

            prediction_columns = Evaluator.get_prediction_columns(df, models)
            print(f"got {len(prediction_columns)} prediction columns for dataset {dataset} and prompt {prompt}", prediction_columns)
            prompt_config = ConfigLoader.get_prompt_config(prompt)
            eval_func, parsing_func = prompt_config["evaluation_function"], prompt_config["parsing_function"]
            


            prediction_df = Evaluator.evaluate_predictions(df, prediction_columns, eval_func, parsing_func, label_col, keep_columns)
            # check if the prediction_df has the same values for majority_vote and unanimous_vote as the reference dataset
         
            

            prediction_df = Evaluator.calculate_votes(prediction_df, prediction_columns)

            # compare with 

            eval_dfs[prompt] = prediction_df
            no_predictions = False

        if no_predictions:
            print(f"No predictions found for dataset {dataset} and prompts {prompts}")
            return None, None

        multiprompt_prediction_df = Evaluator.merge_prompt_evaluations(eval_dfs, keep_columns)

        # print(f"got {len(eval_dfs)} prompt evaluations for dataset {dataset}", eval_dfs.keys())

       
        return multiprompt_prediction_df, eval_dfs
    
    @staticmethod
    def load_data(file_path):
        return pd.read_json(file_path, orient='records', lines=True, dtype=False)
 
    
    @staticmethod
    def word_level_precision_recall(prediction : str, label : str):
        if label.lower() == "russian":
            print("Russian detected")
            print(f"prediction: {prediction}, label: {label}")        
        if not prediction or not label:
            return None, None
        pred_words = set(prediction.lower().split())
        label_words = set(label.lower().split())
        common_words = pred_words.intersection(label_words)
        precision = len(common_words) / len(pred_words) if pred_words else 0
        recall = len(common_words) / len(label_words) if label_words else 0
            
        
        return precision, recall

    @staticmethod
    def get_eval_func(prompt, eval_method):
        if eval_method == "bool":
            return ConfigLoader.get_prompt_config(prompt)["evaluation_function"]
        elif eval_method == "precision_recall":
            if "predict_answer" in prompt: # for predict answer we need to evaluate the precision and recall at the word level
                return Evaluator.word_level_precision_recall
            else: 
                return ConfigLoader.get_prompt_config(prompt)["evaluation_function"]
            
    @staticmethod
    def evaluate_predictions_decision_tree(df, evaluation_df, prompt, model_output_column, eval_method, label_column):
        print(f"Evaluating predictions for prompt {prompt} and model {model_output_column}")
        
        # Preprocess labels and predictions
        df[label_column] = df[label_column].fillna("unanswerable").replace("", "unanswerable")
        
        parsing_func = ConfigLoader.get_prompt_config(prompt)["parsing_function"]
        eval_func = Evaluator.get_eval_func(prompt, eval_method)

        # Parse predictions and create a new column
        parsed_column = f"{model_output_column}_parsed"
        df[parsed_column] = df[model_output_column].apply(lambda x: parsing_func(x) if x is not None else "unanswerable")

        # Replace ";" with " " in both parsed predictions and labels
        df[parsed_column] = df[parsed_column].str.replace(";", " ")
        df[label_column] = df[label_column].str.replace(";", " ")

        # Evaluate the predictions
        if eval_method == "bool":
            evaluation_df[f"{model_output_column}_{prompt}"] = df.apply(lambda row: eval_func(row[parsed_column], row[label_column]), axis=1)
        elif eval_method == "precision_recall":
            print("Evaluating precision and recall")
            if "predict_answer" in prompt:
                precision_recall = df.apply(lambda row: eval_func(row[parsed_column], row[label_column]), axis=1)
                evaluation_df[f"{model_output_column}_{prompt}_precision"] = precision_recall.apply(lambda x: x[0] if x else None)
                evaluation_df[f"{model_output_column}_{prompt}_recall"] = precision_recall.apply(lambda x: x[1] if x else None)
            else:
                evaluation_df[f"{model_output_column}_{prompt}"] = df.apply(lambda row: eval_func(row[parsed_column], row[label_column]), axis=1)

        return evaluation_df

    @staticmethod
    def convert_to_number(x):
        if pd.isna(x):
            return 0.5
        elif x is True:
            return 1
        elif x is False:
            return 0
        elif x is None:
            return 0.5
        elif isinstance(x, float):
            return x
        else:
            print(f"x is not a boolean, None, or NaN or float: {x}")
            return 0.5
    @staticmethod
    def convert_inputs_to_numbers(df, keep_columns):
        
        for column in df.columns:
            if column not in keep_columns:
                df[column] = df[column].apply(lambda x: Evaluator.convert_to_number(x))
        return df
    
    


    @staticmethod
    def create_decision_tree_input( models : list[str] ,prompts :list[str] ,dataset : str, eval_method : str, with_labels : bool = False, ground_truth_column : str = None, verbose : bool = False):
        dataset_config = ConfigLoader.get_dataset_config(dataset)
        id_column = dataset_config["id_column"]
        label_column = dataset_config["label_column"]
        model_output_columns = [ConfigLoader.get_model_config(model)["output_column"] for model in models]
        
        # Load and evaluate predictions
        
        input_df = pd.DataFrame()
        # add id column to the input_df
        input_df[id_column] =  Evaluator.load_data(ConfigLoader.build_model_output_dataset_path(dataset, prompts[0]))[id_column]
        for prompt in prompts:
            print(f"loading predictions for prompt {prompt}")
            prediction_path = ConfigLoader.build_model_output_dataset_path(dataset, prompt)
            df = Evaluator.load_data(prediction_path)
            


            for model, model_output_column in zip(models, model_output_columns): 
                input_df = Evaluator.evaluate_predictions_decision_tree(df, input_df, prompt, model_output_column, eval_method, label_column)
                
        
        # convert inpt_df values to numbers
        input_df = Evaluator.convert_inputs_to_numbers(input_df, [id_column, label_column])
        if verbose:
            for index, row in input_df.iterrows():
                if row[id_column] == "133_2":
                    print(json.dumps(row.to_dict(), indent=4))
        label_df = None
        if with_labels:
            label_df = Evaluator.load_data(ConfigLoader.build_relabeled_evaluation_dataset_path(dataset))

        if ground_truth_column is not None and label_df is not None:
            # filter out the rows that are not in the relabeled_df
            input_df = input_df[input_df[id_column].isin(label_df[id_column])]

            # align the input_df and relabeled_df on the id_column
            input_df = input_df.sort_values(by=id_column)
            label_df = label_df.sort_values(by=id_column)
            
            # to feature matrix and label vector
            X = input_df.drop(columns=[id_column]).values
            if verbose:
                print("X: ", X)
                print("input_df length: ", len(input_df)) 
                print("labeled_df length: ", len(label_df))

            y = label_df[ground_truth_column].apply(lambda x: Evaluator.convert_to_number(x))
            # to numpy array
            X = np.array(X)
            y = np.array(y)
        else:
            X = input_df.drop(columns=[id_column]).values
            X = np.array(X)
            y = None

        return X, y

        
        

        



    

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
        
        return prediction_df

    def calculate_votes(prediction_df: pd.DataFrame, prediction_columns: List[str], invalidate_nan : bool = False) -> pd.DataFrame:
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


        prediction_df['majority_vote'] = prediction_df.apply(majority_vote, axis=1)
        prediction_df['unanimous_vote'] = prediction_df.apply(unanimous_vote, axis=1)
        prediction_df['percentage_vote'] = prediction_df.apply(percentage_vote, axis=1)

        return prediction_df

    def merge_prompt_evaluations(eval_dfs: Dict[str, pd.DataFrame], keep_columns : List[str]) -> pd.DataFrame:
        if not eval_dfs:
            print("No prompt evaluations to merge")
            return pd.DataFrame()
        
        multiprompt_df = next(iter(eval_dfs.values()))[keep_columns].copy()

        for i in range(len(multiprompt_df)):
            rows = {prompt: df.iloc[i] for prompt, df in eval_dfs.items()}

            majority_votes = [row["majority_vote"] for row in rows.values() if row["majority_vote"] is not None]
            multiprompt_df.at[i, "majority_vote"] = (sum(majority_votes) > len(majority_votes) / 2) if majority_votes else None

            unanimous_votes = [row["unanimous_vote"] for row in rows.values() ]
            multiprompt_df.at[i, "unanimous_vote"] = True if all(unanimous_votes) else False if all(vote is False for vote in unanimous_votes) else None

            percentage_votes = [row["percentage_vote"] for row in rows.values() if row["percentage_vote"] is not None]
            multiprompt_df.at[i, "percentage_vote"] = sum(percentage_votes) / len(percentage_votes) if percentage_votes else None

            model_unanimous_votes = [row["unanimous_vote"] for row in rows.values() if row["unanimous_vote"] is not None]
            multiprompt_df.at[i, "model_unanimous_prompt_majority_vote"] = (sum(model_unanimous_votes) > len(model_unanimous_votes) / 2) if model_unanimous_votes else None

            model_majority_votes = [row["majority_vote"] for row in rows.values() if row["majority_vote"] is not None]
            multiprompt_df.at[i, "model_majority_prompt_unanimous_vote"] = all(model_majority_votes) if model_majority_votes else None

        return multiprompt_df

    def save_predictions(df: pd.DataFrame, path: str) -> None:
        try:
            df.to_json(path, orient='records', lines=True)
            print(f"Saved predictions to: {path}")
        except Exception as e:
            print(f"Error saving predictions to {path}: {e}")



def compare_dataframes(ref_df, pred_df, columns_to_check):
    for column in columns_to_check:
        # check datatypes of the columns
        print(f"Comparing {column} values of ref_df and pred_df")
        
        # Create a mismatch mask using only .equals()
        mismatch_mask = ~ref_df[column].equals(pred_df[column])

        

        #for i, row in ref_df.iterrows():
         #   print(f"ref_df: {row[column]}, pred_df: {pred_df.at[i, column]}")
        
      