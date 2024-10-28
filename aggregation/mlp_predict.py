import argparse
import pandas as pd
import numpy as np
from src.ConfigLoader import ConfigLoader
from src.VoteEvaluator import Evaluator
from dotenv import load_dotenv
import os
from joblib import load

load_dotenv()

def load_data(file_path):
    return pd.read_json(file_path, orient='records', lines=True, dtype=False)

def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]

def main():
    parser = argparse.ArgumentParser(description="Predict using a trained MLP classifier on model outputs")
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
    
    # Get features without labels
    X, _ = Evaluator.create_decision_tree_input(models=models, prompts=prompts, dataset=dataset, 
                                                eval_method=eval_method, with_labels=False, verbose=verbose)

    if X is None:
        raise ValueError("X is None")

    # Load the trained model and scaler
    model_path = ConfigLoader.build_classifier_path("mlp", "", [args.model_group, args.prompt_group, args.eval_method, args.id])
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    mlp_classifier, scaler = load(model_path)
    
    # Scale the input features
    X_scaled = scaler.transform(X)
    
    # Make predictions (probabilities)
    y_pred_proba = mlp_classifier.predict_proba(X_scaled)[:, 1]  # Probability of positive class
    
    # Create a DataFrame with the predictions
    io_path = ConfigLoader.build_aggregated_predictions_dataset_path(dataset)
    predictions_df = pd.read_json(io_path, orient="records", lines=True, dtype=False)
    
    predictions_df[model_name] = y_pred_proba
    
    # Save the predictions to the aggregated_predictions dataset
    predictions_df.to_json(io_path, orient='records', lines=True)
    
    print(f"Predictions saved to: {io_path}")
    
    if verbose:
        print("\nPrediction Results:")
        for i, prob in enumerate(y_pred_proba):
            print(f"ID: {predictions_df['id'][i]}, Probability: {prob:.2f}")

if __name__ == "__main__":
    main()
