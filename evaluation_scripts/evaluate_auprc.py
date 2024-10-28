import numpy as np
import argparse
from src.ConfigLoader import ConfigLoader
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()

LABEL_COLUMN = "matches_original"

def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]


def calculate_precision_recall(y_true, y_scores):
    # Combine scores and labels, then sort by score in ascending order
    combined = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    
    total_positives = sum([1 for label in y_true if label == 0])
    precisions, recalls = [], []
    true_positives = 0
    
    for i, (score, label) in enumerate(combined, 1):
        counted = False
        if label == 0:  
            counted = True
            true_positives += 1
        precision = true_positives / i if i > 0 else 0
        recall = true_positives / total_positives if total_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls


def get_column_name(model_group, prompt_group, aggregation_method,eval_method, id, invalidate_nan):
    if "vote" in aggregation_method:
        return aggregation_method + "_"+prompt_group+ "-"+ model_group + ("_no_nan" if invalidate_nan else "")
    
    elif "baseline" in aggregation_method:
        return aggregation_method

    else:
        # ml based method
        classifier_name = ConfigLoader.get_classifier_name(aggregation_method, "", [model_group, prompt_group,eval_method, id])
        return classifier_name
def get_predictions_for_model(df, model_group, prompt_group, aggregation_method,eval_method, id, invalidate_nan):

        return df[get_column_name(model_group, prompt_group, aggregation_method,eval_method, id, invalidate_nan)]


def plot_precision_recall_curves(precision_recall_points, prompt_groups, model_groups, aggregation_methods, eval_methods, ids, invalidate_nan):
    fig, ax = plt.subplots(figsize=(10, 12))  # Increased height to accommodate legend
    for col_name, points in precision_recall_points.items():
        recalls, precisions = zip(*points)
        display_name = input(f"Enter desired display name for '{col_name}' (or press Enter to keep as is): ")
        label = display_name if display_name else col_name
        ax.plot(recalls, precisions, label=f"{label}", linewidth=2.5)  # Increased line width
    
    
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_aspect('equal')  # This ensures the plot is square
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Adjust the plot area to be square and leave room for the legend
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=20)
    filename_parts = [
        'evaluations/precision_recall_curves',
        prompt_groups.replace(",", "-"),
        model_groups.replace(",", "-"),
        aggregation_methods.replace(",", "-"),
        
    ]
    if eval_methods:
        filename_parts.append(eval_methods.replace(",", "-"))
    if ids:
        filename_parts.append(ids.replace(",", "-"))
    if invalidate_nan:
        filename_parts.append("no_nan")
    plt.savefig('_'.join(filename_parts) + '.png', bbox_inches='tight')

    plt.close()

def main():

    parser = argparse.ArgumentParser(description="Evaluate the AUC-PR of different methods ")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--eval_methods",  help="Evaluation method")
    parser.add_argument("--model_groups", required=True, help="Name of the model group")
    parser.add_argument("--prompt_groups", required=True, help="Name of the prompt group")
    parser.add_argument("--ids",  help="Identifier for the model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--aggregation_methods", required=True, help="Name of the aggregation method")
    parser.add_argument("--invalidate_nan", action="store_true", help="Invalidate nan values")


    args = parser.parse_args()

    #split model groups by comma
    model_groups = args.model_groups.split(",") 
    prompt_groups = args.prompt_groups.split(",") 
    ids = args.ids.split(",") if args.ids else [["default"]]
    aggregation_methods = args.aggregation_methods.split(",")
    eval_methods = args.eval_methods.split(",") if args.eval_methods else ["default"]

    dataset_path = ConfigLoader.build_aggregated_predictions_dataset_path(args.dataset)
    df = pd.read_json(dataset_path, orient="records", lines=True, dtype=False)

    label_dataset_path = ConfigLoader.build_relabeled_evaluation_dataset_path(args.dataset)
    labels_df = pd.read_json(label_dataset_path, orient="records", lines=True, dtype=False)

    # filter the df to only include instances that are in the labels_df
    df = df[df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]].isin(labels_df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]])]
    # order both df by the id column
    df = df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])
    labels_df = labels_df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])

    if args.invalidate_nan:
        cols = []
        for model_group in model_groups:
            for prompt_group in prompt_groups:
                # load the aggregated predictions for this model group and prompt group
                for aggregation_method in aggregation_methods:
                    for eval_method in eval_methods:
                        for id in ids:   
                            cols.append(get_column_name(model_group, prompt_group, aggregation_method, eval_method, id, args.invalidate_nan))

        cols = list(set(cols))
        def filter_nan(row):
            return not any(pd.isna(row[col]) or row[col] == None for col in cols)
        
        df = df[df.apply(filter_nan, axis=1)]
        
        # Filter labels_df to match the remaining rows in df
        labels_df = labels_df[labels_df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]].isin(df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]])]

    
    precision_recall_points = {}
    auc_pr_scores = {}

    for model_group in model_groups:
        for prompt_group in prompt_groups:
            # load the aggregated predictions for this model group and prompt group
            for aggregation_method in aggregation_methods:
                for eval_method in eval_methods:
                    for id in ids:

                        print("Evaluating model group:", model_group, " prompt group:", prompt_group, " aggregation method:", aggregation_method, " eval method:", eval_method, " id:", id)
                        

                        predictions = get_predictions_for_model(df, model_group, prompt_group, aggregation_method,eval_method, id, args.invalidate_nan)  
                        labels = labels_df[LABEL_COLUMN].map(lambda x: 1 if x == True else 0).to_numpy()
                        print(labels)
                        
                        predictions = predictions.to_numpy() # get scores as numpy array
                        
                        precisions, recalls = calculate_precision_recall(labels, predictions)

                        col_name = get_column_name(model_group, prompt_group, aggregation_method,eval_method, id, args.invalidate_nan)
                        auc_score = np.trapz(precisions, recalls)
                        auc_pr_scores[col_name] = auc_score

                        # Store precision-recall points for plotting
                        precision_recall_points[col_name] = list(zip(recalls, precisions))

    # Plot precision-recall curves
    plot_precision_recall_curves(precision_recall_points, args.prompt_groups, args.model_groups, args.aggregation_methods, args.eval_methods, args.ids, args.invalidate_nan)

    # Print AUC-PR scores
    for col_name, auc_score in auc_pr_scores.items():
        print(f"AUC-PR score for {col_name}: {auc_score:.4f}")

if __name__ == "__main__":
    main()
