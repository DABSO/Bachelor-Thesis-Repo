import argparse
import pandas as pd
import numpy as np
from src.ArgParsers.EvaluationArgParser import EvaluationArgParser
from src.ConfigLoader import ConfigLoader
from src.VoteEvaluator import Evaluator
from dotenv import load_dotenv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from joblib import dump
import optuna

load_dotenv()

LABEL_COLUMN = "matches_original"

def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]

def create_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 5, 15)
        layers.append(n_units)
    
    return MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
        solver='adam',
        alpha=trial.suggest_loguniform('alpha', 1e-5, 1e-2),
        max_iter=1000,
        random_state=42
    )

def objective(trial, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = create_model(trial)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict_proba(X_val_scaled)[:, 1] # Get the probability of the positive class
    y_pred = np.where(y_pred > 0.5, 1, 0) # Convert the probability to a binary prediction

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    f1_negative = 2 * tn / (2 * tn + fp + fn)
    return f1_negative

def optimize_mlp(X, y, n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    return best_params, best_score

def train_best_mlp(X, y, best_params):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    layers = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=best_params['activation'],
        solver='adam',
        alpha=best_params['alpha'],
        max_iter=500,
        random_state=42
    )
    
    mlp.fit(X_scaled, y)
    return mlp, scaler

def main():
    parser = argparse.ArgumentParser(description="Train a Multi-Layer Perceptron classifier on model outputs")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--eval_method", choices=['bool', 'precision_recall'], default='bool', help="Evaluation method")
    parser.add_argument("--model_group", required=True, help="Name of the model group")
    parser.add_argument("--prompt_group", required=True, help="Name of the prompt group")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--id", required=True, help="Identifier for the model")
    args = parser.parse_args()

    dataset = args.dataset
    eval_method = args.eval_method
    verbose = args.verbose
    prompts = get_prompts_in_group(args.prompt_group)
    print("prompts: ", prompts)
    models = get_models_in_group(args.model_group)
    print("models: ", models)
    
    X, y = Evaluator.create_decision_tree_input(models=models, prompts=prompts, dataset=dataset, eval_method=eval_method, with_labels=True, ground_truth_column=LABEL_COLUMN, verbose=verbose)

    if X is None or y is None:
        raise ValueError("X or y is None")

    # Perform hyperparameter optimization
    best_params_list = []
    best_scores_list = []
    
    for i in range(3):
        print(f"\nOptimization trial {i+1}/3")
        best_params, best_score = optimize_mlp(X, y)
        best_params_list.append(best_params)
        best_scores_list.append(best_score)
        print(f"Best parameters: {best_params}")
        print(f"Best F1 score: {best_score:.4f}")

    # Select the best parameters based on average F1 score
    best_index = np.argmax(best_scores_list)
    best_params = best_params_list[best_index]
    print(f"\nBest overall parameters: {best_params}")
    print(f"Average F1 score: {np.mean(best_scores_list):.4f}")

    # Train the final model with the best parameters
    mlp_classifier, scaler = train_best_mlp(X, y, best_params)
    
    # Make predictions and evaluate
    X_scaled = scaler.transform(X)
    y_pred_proba = mlp_classifier.predict_proba(X_scaled)[:, 1]
    y_pred = mlp_classifier.predict(X_scaled)
    
    # Print results
    print("\nPrediction Results:")
    for i, (true, pred, prob) in enumerate(zip(y, y_pred, y_pred_proba)):
        correct = "Correct" if true == pred else "Incorrect"
        print(f"ID:  True: {true:.2f}, Predicted: {pred:.2f}, Probability: {prob:.2f}, {correct}")
    
    # Calculate and print accuracy and F1 score for negative class
    accuracy = (y == y_pred).mean()
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    f1_negative = 2 * tn / (2 * tn + fp + fn)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"F1 Score for Negative Class: {f1_negative:.4f}")

    # Store the model and scaler
    model_path = ConfigLoader.build_classifier_path("mlp", "", [args.model_group, args.prompt_group, args.eval_method, args.id])
    dump((mlp_classifier, scaler), model_path)

if __name__ == "__main__":
    main()
