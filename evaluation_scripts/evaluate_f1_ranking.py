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

def calculate_f1_scores(y_true, y_scores):
    thresholds = np.arange(0, 1.01, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        # Negative class
        precision_neg = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else None
        recall_neg = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else None
        f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg is not None and recall_neg is not None and precision_neg + recall_neg > 0) else None
        
        # Print metrics
        precision_str = f"{precision_neg:.3f}" if precision_neg is not None else "None"
        recall_str = f"{recall_neg:.3f}" if recall_neg is not None else "None"
        f1_str = f"{f1_neg:.3f}" if f1_neg is not None else "None"
        
        print(f"neg_precision: {precision_str}, neg_recall: {recall_str}, neg_f1: {f1_str}, threshold: {threshold:.3f}")
        
        scores.append((f1_neg, precision_neg, recall_neg))
    
    return thresholds, scores

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

def plot_f1_curves(points, prompt_groups, model_groups, aggregation_methods, eval_methods, ids, add_precision_recall=False):
    fig, ax = plt.subplots(figsize=(10, 12))
    
    for i, (col_name, points) in enumerate(points.items()):
        thresholds, f1_scores, precision_scores, recall_scores = points
        display_name = input(f"Enter desired display name for '{col_name}' (or press Enter to keep as is): ")
        label = display_name if display_name else col_name
        
        # Filter out None values
        valid_points = [(t, f, p, r) for t, f, p, r in zip(thresholds, f1_scores, precision_scores, recall_scores) if f is not None]
        if not valid_points:
            print(f"Warning: No valid points for {label}")
            continue
        
        valid_thresholds, valid_f1, valid_precision, valid_recall = zip(*valid_points)
        
        f1_line, = ax.plot(valid_thresholds, valid_f1, label=f"{label} (F1)", linewidth=2.5)
        if add_precision_recall:
            color = f1_line.get_color()
            ax.plot(valid_thresholds, valid_precision, label=f"{label} (Precision)", linestyle=':', color=color, linewidth=2.5)
            ax.plot(valid_thresholds, valid_recall, label=f"{label} (Recall)", linestyle='--', color=color, linewidth=2.5)
    # Adjust the plot area to be square and leave room for the legend
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlabel('Threshold', fontsize=20)
    ax.set_ylabel('F1 Score' if not add_precision_recall else 'Scores', fontsize=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_aspect('equal')
    
    # Adjust the plot area to be square and leave room for the legend
    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=20)
    filename_parts = [
        'evaluations/f1_curves',
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
    parser = argparse.ArgumentParser(description="Evaluate F1 scores at different thresholds for model outputs")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--eval_methods", help="Evaluation method")
    parser.add_argument("--model_groups", required=True, help="Name of the model group")
    parser.add_argument("--prompt_groups", required=True, help="Name of the prompt group")
    parser.add_argument("--ids", help="Identifier for the model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--aggregation_methods", required=True, help="Name of the aggregation method")
    parser.add_argument("--add_precision_recall", action="store_true", help="Add precision and recall to the plot")

    args = parser.parse_args()

    model_groups = args.model_groups.split(",")
    prompt_groups = args.prompt_groups.split(",")
    ids = args.ids.split(",") if args.ids else ["default"]
    aggregation_methods = args.aggregation_methods.split(",")
    eval_methods = args.eval_methods.split(",") if args.eval_methods else ["default"]

    add_precision_recall = args.add_precision_recall or False

    dataset_path = ConfigLoader.get_aggregated_predictions_dataset_path(args.dataset)
    df = pd.read_json(dataset_path, orient="records", lines=True, dtype=False)

    label_dataset_path = ConfigLoader.get_relabeled_evaluation_dataset_path(args.dataset)
    labels_df = pd.read_json(label_dataset_path, orient="records", lines=True, dtype=False)

    df = df[df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]].isin(labels_df[ConfigLoader.get_dataset_config(args.dataset)["id_column"]])]
    df = df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])
    labels_df = labels_df.sort_values(by=ConfigLoader.get_dataset_config(args.dataset)["id_column"])

    graph_points = {}
    max_f1_scores = {}
    recalls_at_threshold = {}
    precisions_at_threshold = {}

    for model_group in model_groups:
        for prompt_group in prompt_groups:
            for aggregation_method in aggregation_methods:
                for eval_method in eval_methods:
                    for id in ids:
                        print(f"Evaluating model group: {model_group}, prompt group: {prompt_group}, aggregation method: {aggregation_method}, eval method: {eval_method}, id: {id}")

                        predictions = get_predictions_for_model(df, model_group, prompt_group, aggregation_method, eval_method, id)
                        labels = labels_df[LABEL_COLUMN].map(lambda x: 1 if x == True else 0).to_numpy()
                        
                        predictions = predictions.to_numpy()
                        
                        thresholds, scores = calculate_f1_scores(labels, predictions)

                        col_name = get_column_name(model_group, prompt_group, aggregation_method, eval_method, id)
                        
                        max_f1_score = np.max([score[0] for score in scores if score[0] is not None])
                        max_f1_scores[col_name] = max_f1_score

                        recalls_at_threshold[col_name] = [score[2] for score in scores if score[2] is not None]
                        precisions_at_threshold[col_name] = [score[1] for score in scores if score[1] is not None]

                        graph_points[col_name] = (thresholds, [score[0] for score in scores], [score[1] for score in scores], [score[2] for score in scores])
    for col_name in max_f1_scores.keys():
        max_f1 = max_f1_scores[col_name]
        
        # Find smallest threshold where recall is 1
        recall_threshold = next((thresholds[i] for i, recall in enumerate(recalls_at_threshold[col_name]) if recall == 1), None)
        
        # Find largest threshold where precision is 1
        precision_threshold = next((thresholds[-i-1] for i, precision in enumerate(reversed(precisions_at_threshold[col_name])) if precision == 1), None)
        
        print(f"Metrics for {col_name}:")
        print(f"  Max F1 score: {max_f1:.4f}")
        print(f"  Smallest threshold where recall is 1: {recall_threshold:.4f}" if recall_threshold is not None else "  Recall never reaches 1")
        print(f"  Largest threshold where precision is 1: {precision_threshold:.4f}" if precision_threshold is not None else "  Precision never reaches 1")
        


    plot_f1_curves(graph_points, args.prompt_groups, args.model_groups, args.aggregation_methods, args.eval_methods, args.ids, add_precision_recall)

    
if __name__ == "__main__":
    main()
