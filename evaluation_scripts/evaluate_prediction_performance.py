import pandas as pd
from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv
import argparse
from collections import Counter
import copy
from difflib import SequenceMatcher
from typing import List, Tuple

def relabeled_exact_match(relabeled_row, predicted_row, prediction_column, label_column):
    

    if predicted_row[prediction_column] == "" and relabeled_row["unanswerable"] is True:
        return True
    elif predicted_row[prediction_column] != "" and relabeled_row["unanswerable"] is True and relabeled_row[label_column] == {}:
        return False

    if predicted_row[prediction_column] is None:
        return False
    original_labels = [l.strip() for l in predicted_row[prediction_column].split(";")]    
    for k, v in relabeled_row[label_column].items():
        # check if there is an exact match 
        found_match = False
        if any(label == k.strip() for label in original_labels ):
            found_match = True
            #remove the matched label from the original labels 
            original_labels = [label for label in original_labels if label != k.strip()]
        
        if not found_match: # check the alternatives 
            for item in v["alternative_labels"]:
                # check if an alternative matches one of the origial labels
                if any(item.strip() == label for label in original_labels):
                    found_match = True
                    #remove the matched label from the original labels 
                    original_labels = [label for label in original_labels if label != item.strip()]

        if not found_match and v["type"] != "label_candidate": # there is a label in the relabeled data that does not match the original and is not a label candidate
            return False
    if original_labels: # there are labels in the original data that are not in the relabeled data
        return False
    
    return True

def word_wise_f1(pred_words: List[str], true_words: List[str]) -> float:
    common_words = set(pred_words) & set(true_words)
    precision = len(common_words) / len(pred_words) if pred_words else 0
    recall = len(common_words) / len(true_words) if true_words else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def relabeled_f1_score(relabeled_row, predicted_row, prediction_column, label_column):
    if predicted_row[prediction_column] == "" and relabeled_row["unanswerable"] is True:
        return 1
    elif predicted_row[prediction_column] == "" and relabeled_row["unanswerable"] is False:
        return 0
    elif predicted_row[prediction_column] is None:
        return 0
    
    predicted_labels = [l.strip() for l in predicted_row[prediction_column].split(";")]
    true_labels = relabeled_row[label_column]
    
    total_f1 = 0
    matched_true_labels = set()
    
    for pred_label in predicted_labels:
        pred_words = pred_label.lower().split()
        best_match = None
        best_f1 = 0
        
        for true_label, info in true_labels.items():
            if true_label in matched_true_labels:
                continue
            true_words = true_label.lower().split()
            f1 = word_wise_f1(pred_words, true_words)
            
            for alt_label in info["alternative_labels"]:
                alt_words = alt_label.lower().split()
                alt_f1 = word_wise_f1(pred_words, alt_words)
                if alt_f1 > f1:
                    f1 = alt_f1
                    true_words = alt_words
            
            if f1 > best_f1:
                best_f1 = f1
                best_match = true_label
        
        if best_match:
            matched_true_labels.add(best_match)
            total_f1 += best_f1
    
    # Handle unmatched true labels that are not label candidates
    for label, info in true_labels.items():
        if label not in matched_true_labels and info["type"] != "label_candidate":
            total_f1 += 0  # Adding 0 for unmatched non-candidate labels
    
    # Calculate average F1 score
    num_labels = len(predicted_labels) + len([l for l, info in true_labels.items() if info["type"] != "label_candidate" and l not in matched_true_labels])
    average_f1 = total_f1 / num_labels if num_labels > 0 else 0
    
    return average_f1


def evaluate_prediction_performance_relabeled_f1(prediction_df : pd.DataFrame, prediction_column : str, label_column : str, label_df : pd.DataFrame, id_column : str):
    sum_f1 = 0
    total_labels = 0
    for index, row in prediction_df.iterrows():
        sum_f1 += relabeled_f1_score(label_df.loc[index], row, prediction_column, label_column)
        total_labels += 1
    average_f1 = sum_f1 / total_labels
    return average_f1

def parse_predictions(prediction_df : pd.DataFrame, prediction_column : str, parsing_function : callable):
    prediction_df[prediction_column] = prediction_df[prediction_column].apply(lambda x: parsing_function(x) if x is not None else None)
    return prediction_df

def evaluate_em(string: str, label : str) -> bool:
    if string is None:
        return None

    # split the label and string into 2 lists of answers
    predictions = string.split(";")
    predictions = [prediction.strip() for prediction in predictions]

    labels = label.split(";")
    labels = [l.strip() for l in labels]

    # check if the predictions match the labels exactly
    is_match = check_exact_match(predictions, labels)
    return is_match

@staticmethod
def check_exact_match(arr1 , arr2):
    if arr1 == arr2:
        return True
    elif arr1 is None or arr2 is None: 
        return False
    elif Counter(arr1) == Counter(arr2):
        return True
    else:
        return False

def evaluate_prediction_performance_em(prediction_df : pd.DataFrame, prediction_column : str, label_column : str):
    true_predictions = 0
    false_predictions = 0
    for index, row in prediction_df.iterrows():
        if evaluate_em(row[prediction_column], row[label_column]):
            true_predictions += 1
        else:
            false_predictions += 1
    return true_predictions, false_predictions

def evaluate_prediction_performance_relabeled_em(prediction_df : pd.DataFrame, prediction_column : str, label_column : str, label_df : pd.DataFrame, id_column : str):
    
    true_predictions = 0
    false_predictions = 0
    for index, row in prediction_df.iterrows():
        if relabeled_exact_match(label_df.loc[index], row, prediction_column, label_column):
            true_predictions += 1
        else:
            false_predictions += 1
    return true_predictions, false_predictions

def evaluate_f1(prediction: str, label: str) -> float:
    if prediction is None:
        return 0.0

    # Split the prediction and label into lists of entities
    pred_entities = [entity.strip() for entity in prediction.split(";")]
    true_entities = [entity.strip() for entity in label.split(";")]

    total_f1 = 0
    matched_true_entities = set()

    for pred_entity in pred_entities:
        pred_words = pred_entity.lower().split()
        best_match = None
        best_f1 = 0

        for true_entity in true_entities:
            if true_entity in matched_true_entities:
                continue
            true_words = true_entity.lower().split()
            f1 = word_wise_f1(pred_words, true_words)

            if f1 > best_f1:
                best_f1 = f1
                best_match = true_entity

        if best_match:
            matched_true_entities.add(best_match)
            total_f1 += best_f1

    # Handle unmatched true entities
    unmatched_true_entities = len(true_entities) - len(matched_true_entities)
    total_f1 += 0 * unmatched_true_entities  # Adding 0 for unmatched true entities

    # Calculate average F1 score
    num_entities = max(len(pred_entities), len(true_entities))
    average_f1 = total_f1 / num_entities if num_entities > 0 else 0

    return average_f1

def evaluate_prediction_performance_f1(prediction_df: pd.DataFrame, prediction_column: str, label_column: str) -> float:
    total_f1 = 0
    num_samples = 0

    for index, row in prediction_df.iterrows():
        f1_score = evaluate_f1(row[prediction_column], row[label_column])
        total_f1 += f1_score
        num_samples += 1

    average_f1 = total_f1 / num_samples if num_samples > 0 else 0
    return average_f1
def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--model_group", required=True, help="Name of the model ")
    parser.add_argument("--prompt", required=True, help="Name of the prompt")
    parser.add_argument("--use_relabeled", action="store_true", help="Use relabeled data")
    parser.add_argument("--metric", required=True, help="Metric to use", choices=["em", "f1"])

    args = parser.parse_args()
    dataset = args.dataset
    model_group = args.model_group
    prompt = args.prompt
    use_relabeled = args.use_relabeled or False
    metric = args.metric
    dataset_config = ConfigLoader.get_dataset_config(dataset)
    id_column = dataset_config["id_column"]
    label_column = dataset_config["label_column"]
    prompt_config = ConfigLoader.get_prompt_config(prompt)
    parsing_function = prompt_config["parsing_function"]
    for model in get_models_in_group(model_group):
        model_config = ConfigLoader.get_model_config(model)
        model_output_column = model_config["output_column"]
        prediction_df = pd.read_json(ConfigLoader.build_model_output_dataset_path(dataset, prompt), lines=True, orient="records", dtype=False)
        prediction_df = parse_predictions(prediction_df, model_output_column, parsing_function)
        if use_relabeled:
            labeled_dataset_path = ConfigLoader.build_relabeled_evaluation_dataset_path(dataset)
            label_df = pd.read_json(labeled_dataset_path, lines=True, orient="records", dtype=False)
            # filter the prediction df to only include the rows that are in the label df
            prediction_df = prediction_df[prediction_df[id_column].isin(label_df[id_column])]
            # order by id column
            prediction_df = prediction_df.sort_values(by=id_column).reset_index(drop=True)
            label_df = label_df.sort_values(by=id_column).reset_index(drop=True)
            if metric == "em":
                true_predictions, false_predictions = evaluate_prediction_performance_relabeled_em(prediction_df, model_output_column, label_column, label_df, id_column)
                print(f"{model} Accuracy Relabeled: {true_predictions / (true_predictions + false_predictions)}")
            elif metric == "f1":
                average_f1 = evaluate_prediction_performance_relabeled_f1(prediction_df, model_output_column, label_column, label_df, id_column)
                print(f"{model}Average F1 Accuracy Relabeled: {average_f1}")
        else:
            if metric == "em":
                true_predictions, false_predictions = evaluate_prediction_performance_em(prediction_df, model_output_column, label_column)
                print(f"{model} Accuracy Original: {true_predictions / (true_predictions + false_predictions)}")
            elif metric == "f1":
                average_f1 = evaluate_prediction_performance_f1(prediction_df, model_output_column, label_column)
                print(f"{model} Average F1 Accuracy Original: {average_f1}")



if __name__ == "__main__":  
    load_dotenv()
    main()

