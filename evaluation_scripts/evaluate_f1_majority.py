import numpy as np
import argparse
from src.ConfigLoader import ConfigLoader
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()

LABEL_COLUMN = "matches_original"
THRESHOLD = 0.5

def get_column_name(model_group, prompt_group, aggregation_method, eval_method, id):
    if "vote" in aggregation_method:
        return aggregation_method + "_" + prompt_group + "-" + model_group
    elif "baseline" in aggregation_method:
        return aggregation_method
    else:
        classifier_name = ConfigLoader.get_classifier_name(aggregation_method, "", [model_group, prompt_group, eval_method, id])
        return classifier_name

def get_predictions_for_model(df, model_group, prompt_group, aggregation_method, eval_method, id):
    return df[get_column_name(model_group, prompt_group, aggregation_method, eval_method, id)]

def calculate_f1_score(y_true, y_scores, threshold=THRESHOLD):
    y_pred = (y_scores >= threshold).astype(int)
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
    recall = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall

def plot_f1_bar(f1_scores, prompt_groups, model_groups, aggregation_methods, eval_methods, ids):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = list(f1_scores.keys())
    display_labels = []
    for label in labels:
        print(f"Current label: {label}")
        display_name = input(f"Enter desired display name for '{label}' (or press Enter to keep as is): ")
        display_labels.append(display_name if display_name else label)
    
    f1_values = [score[0] for score in f1_scores.values()]
    precision_values = [score[1] for score in f1_scores.values()]
    recall_values = [score[2] for score in f1_scores.values()]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, f1_values, width, label='F1 Score')
    ax.bar(x, precision_values, width, label='Precision')
    ax.bar(x + width, recall_values, width, label='Recall')
    
    ax.set_ylabel('Scores', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.legend(fontsize=20)
    
    plt.tight_layout()
    filename_parts = [
        'evaluations/f1_bar',
        prompt_groups.replace(",", "-"),
        model_groups.replace(",", "-"),
        aggregation_methods.replace(",", "-")
    ]
    if eval_methods:
        filename_parts.append(eval_methods.replace(",", "-"))
    if ids:
        filename_parts.append(ids.replace(",", "-"))
    plt.savefig('_'.join(filename_parts) + '.png', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate F1 scores at threshold 0.5 for model outputs")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--eval_methods", help="Evaluation method")
    parser.add_argument("--model_groups", required=True, help="Name of the model group")
    parser.add_argument("--prompt_groups", required=True, help="Name of the prompt group")
    parser.add_argument("--ids", help="Identifier for the model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--aggregation_methods", required=True, help="Name of the aggregation method")

    args = parser.parse_args()

    model_groups = args.model_groups.split(",")
    prompt_groups = args.prompt_groups.split(",")
    ids = args.ids.split(",") if args.ids else ["default"]
    aggregation_methods = args.aggregation_methods.split(",")
    eval_methods = args.eval_methods.split(",") if args.eval_methods else ["default"]

    dataset_path = ConfigLoader.build_aggregated_predictions_dataset_path(args.dataset)
    df = pd.read_json(dataset_path, orient="records", lines=True, dtype=False)

    label_dataset_path = ConfigLoader.build_relabeled_evaluation_dataset_path(args.dataset)
    labels_df = pd.read_json(label_dataset_path, orient="records", lines=True, dtype=False)

    df = df[df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]].isin(labels_df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]])]
    df = df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])
    labels_df = labels_df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])

    f1_scores = {}

    for model_group in model_groups:
        for prompt_group in prompt_groups:
            for aggregation_method in aggregation_methods:
                for eval_method in eval_methods:
                    for id in ids:
                        col_name = get_column_name(model_group, prompt_group, aggregation_method, eval_method, id)
                        if col_name in f1_scores:
                            continue
                        print(f"Evaluating model group: {model_group}, prompt group: {prompt_group}, aggregation method: {aggregation_method}, eval method: {eval_method}, id: {id}")

                        predictions = get_predictions_for_model(df, model_group, prompt_group, aggregation_method, eval_method, id)
                        labels = labels_df[LABEL_COLUMN].map(lambda x: 1 if x == True else 0).to_numpy()
                        
                        predictions = predictions.to_numpy()
                        
                        f1, precision, recall = calculate_f1_score(labels, predictions)

                        
                        f1_scores[col_name] = (f1, precision, recall)

                        print(f"F1 score for {col_name}: {f1:.4f}")
                        print(f"Precision for {col_name}: {precision:.4f}")
                        print(f"Recall for {col_name}: {recall:.4f}")
                        print()

    plot_f1_bar(f1_scores, args.prompt_groups, args.model_groups, args.aggregation_methods, args.eval_methods, args.ids)

if __name__ == "__main__":
    main()