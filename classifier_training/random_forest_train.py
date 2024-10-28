import argparse
from src.ArgParsers.EvaluationArgParser import EvaluationArgParser
import pandas as pd
import numpy as np
from src.ConfigLoader import ConfigLoader
from src.VoteEvaluator import Evaluator
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from joblib import dump

load_dotenv()

LABEL_COLUMN = "matches_original"

def load_data(file_path):
    return pd.read_json(file_path, orient='records', lines=True, dtype=False)


def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]

def train_and_evaluate_random_forest(X, y, n_splits=3, n_iter=100):
    # Define the hyperparameter search space
    param_dist = {
        'n_estimators': [10, 20, 50, 100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    f1_scores = []
    best_models = []

    for i in range(n_splits):
        # Create a new random split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42+i)

        # Initialize RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)

        # Function to evaluate model performance
        def negative_f1_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return f1_score(y, y_pred, pos_label=0, average='binary')

        # Perform RandomizedSearchCV
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                       n_iter=n_iter, cv=3, verbose=2,
                                       n_jobs=-1, scoring=negative_f1_scorer)

        # Fit the random search model
        rf_random.fit(X_train, y_train)

        # Get the best model
        best_rf = rf_random.best_estimator_

        # Evaluate on val set
        y_pred = best_rf.predict(X_val)
        negative_f1 = f1_score(y_val, y_pred, pos_label=0, average='binary')

        f1_scores.append(negative_f1)
        best_models.append(best_rf)

        print(f"Split {i+1} - F1 Score for Negative Class: {negative_f1:.4f}")

    # Calculate average F1 score
    avg_f1_score = np.mean(f1_scores)
    print(f"\nAverage F1 Score for Negative Class: {avg_f1_score:.4f}")

    # Return the model with the best F1 score
    best_model_index = np.argmax(f1_scores)
    return best_models[best_model_index], avg_f1_score



def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier on model outputs")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--eval_method", choices=['bool', 'precision_recall'], default='bool', help="Evaluation method")
    parser.add_argument("--model_group", required=True, help="Name of the model group")
    parser.add_argument("--prompt_group", required=True, help="Name of the prompt group")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--id", required=True, help="Identifier for the model")
    parser.add_argument("--exclude_al", type=str, help="Exclude active learning data from a certain active learning type")
    args = parser.parse_args()

    dataset = args.dataset
    eval_method = args.eval_method
    verbose = args.verbose
     # Get non-legacy prompt names 
    prompts = get_prompts_in_group(args.prompt_group)
    print("prompts: ", prompts)
    models = get_models_in_group(args.model_group)
    print("models: ", models)
    
    X, y = Evaluator.create_decision_tree_input( models=models, prompts=prompts, dataset=dataset, eval_method=eval_method, with_labels=True, ground_truth_column=LABEL_COLUMN, verbose=verbose)

    if X is None or y is None:
        raise ValueError("X or y is None")
    


    # Train and evaluate the random forest with multiple splits
    best_rf_classifier, avg_f1_score = train_and_evaluate_random_forest(X, y, n_splits=1)

    # Make predictions on the entire dataset
    y_pred = best_rf_classifier.predict(X)
    y_pred_proba = best_rf_classifier.predict_proba(X)[:, 1]

    # Print results
    print("\nPrediction Results (sample):")
    for i in range(min(10, len(y))):  # Print first 10 results
        correct = "Correct" if y[i] == y_pred[i] else "Incorrect"
        print(f"ID: {i}, True: {y[i]:.2f}, Predicted: {y_pred[i]:.2f}, Probability: {y_pred_proba[i]:.2f}, {correct}")
    
    print(f"\nBest Model Average F1 Score for Negative Class: {avg_f1_score:.4f}")
    
    # Calculate and print accuracy
    accuracy = (y == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.2f}")
    #store the model
    model_path = ConfigLoader.build_classifier_path("random_forest", "",[args.model_group, args.prompt_group, args.eval_method, args.id])
    dump(best_rf_classifier, model_path)

if __name__ == "__main__":
    main()


