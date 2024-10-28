# find optimal ensemble
from src.VoteEvaluator import Evaluator
from src.ArgParsers.EvaluationArgParser import EvaluationArgParser
from src.ConfigLoader import ConfigLoader
from src.PerformanceTest import PerformanceTest
import itertools
from dotenv import load_dotenv
import pandas as pd
import argparse
load_dotenv()

def get_models_in_group(model_group):
    return [model for model in ConfigLoader.load_available_models() if model_group in ConfigLoader.get_model_config(model)["groups"]]

def get_prompts_in_group(prompt_group):
    return [prompt for prompt in ConfigLoader.load_available_prompts() if prompt_group in ConfigLoader.get_prompt_config(prompt)["groups"]]


def load_prompts(prompt_groups):
    prompts = []
    for prompt_group in prompt_groups:
        prompts += get_prompts_in_group(prompt_group)
    prompts = list(set(prompts))
    return prompts

def load_models(model_groups):
    models = []
    for model_group in model_groups:
        models += get_models_in_group(model_group)
    models = list(set(models))
    return models


def main():
    datasets = ConfigLoader.load_available_datasets()
   

    parser = argparse.ArgumentParser(description="Find the optimal ensemble of models and prompts")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--model_groups", required=True, help="Name of the model groups")
    parser.add_argument("--prompt_groups", required=True, help="Name of the prompt groups")

    args = parser.parse_args()
    dataset = args.dataset
    id_column = ConfigLoader.get_dataset_config(dataset)["id_column"]
    model_groups = args.model_groups.split(",")
    prompt_groups = args.prompt_groups.split(",")

    
    models = load_models(model_groups)
    prompts = load_prompts(prompt_groups)
    
    
    
    label_df = pd.read_json(ConfigLoader.build_relabeled_evaluation_dataset_path(dataset), lines=True, orient="records", dtype=False, convert_axes=False)

    majority_vote_performances_df = pd.DataFrame()
    unanimous_vote_performances_df = pd.DataFrame()
    percentage_treshold_performces_df = pd.DataFrame()
    percentage_treshold_0_5_performces_df = pd.DataFrame()
    auc_pr_df = pd.DataFrame()
    # get all subsets of the models
    print(models)
    subsets = list(itertools.chain(*[itertools.combinations(models, i) for i in range(1, len(models) + 1)]))
    prompt_subsets = list(itertools.chain(*[itertools.combinations(prompts, i) for i in range(1, len(prompts) + 1)]))
    print(subsets)
    print(f"searching best ensemble of {len(subsets) * len(prompt_subsets)} possible subsets of models and prompts  for dataset {dataset}")
    print("prompt subsets", prompt_subsets) 
    
    for subset in subsets:
        
        for prompt_subset in prompt_subsets:
            
            print(subset, prompt_subset)

            merged_df, prompt_dfs = Evaluator.evaluate(list(subset), dataset, list(prompt_subset))
            if merged_df is None:
                continue
            if prompt_dfs is None:
                continue
            
            performance_test = PerformanceTest(merged_df, label_df, id_column)
            

            ensemble_majority_vote_performance_metrics = performance_test.evaluate_votes("majority_vote")
            ensemble_unanimous_vote_performance_metrics = performance_test.evaluate_votes("unanimous_vote")
            ensemble_percentage_treshold_performance_df = performance_test.evaluate_percentage_vote()

            ensemble_auc_pr = performance_test.evaluate_auc_pr("percentage_vote")
            auc_pr_df = auc_pr_df._append({
                "prompt_group": ",".join(prompt_subset),
                "ensemble": ",".join(subset),
                "auc_pr_df": ensemble_auc_pr
            }, ignore_index=True)

            print("performance evaluated")
            print("majority_vote metrics",ensemble_majority_vote_performance_metrics)
            print("unanimous_vote metrics",ensemble_unanimous_vote_performance_metrics)

            # get the row with the best f1 score from the percentage treshold performance df
            best_f1_score_row = ensemble_percentage_treshold_performance_df.sort_values(by="f1_score", ascending=False).iloc[0].to_dict()
            f1_score_0_5_row = ensemble_percentage_treshold_performance_df[ensemble_percentage_treshold_performance_df["threshold"] == 0.5].iloc[0].to_dict()
            print("best f1 score row",best_f1_score_row)

            percentage_treshold_performces_df = percentage_treshold_performces_df._append({
                "prompt_group": ",".join(prompt_subset),
                "ensemble": ",".join(subset),
                **best_f1_score_row,
               
            }, ignore_index=True
            )

            percentage_treshold_0_5_performces_df = percentage_treshold_0_5_performces_df._append({
                "prompt_group": ",".join(prompt_subset),
                "ensemble": ",".join(subset),
                **f1_score_0_5_row,
            }, ignore_index=True
            )

            majority_vote_performances_df = majority_vote_performances_df._append({
                "prompt_group": ",".join(prompt_subset),
                "ensemble": ",".join(subset),
                **ensemble_majority_vote_performance_metrics
            }, ignore_index=True)


            unanimous_vote_performances_df = unanimous_vote_performances_df._append({
                "ensemble" : ",".join(subset),
                "prompt_group": ",".join(prompt_subset),
                **ensemble_unanimous_vote_performance_metrics
            }, ignore_index=True)
            print("-----------------------------")
    base_name =  args.prompt_groups.replace(",", "-") + "_" + args.model_groups.replace(",", "-")
    majority_vote_performances_df.to_json(ConfigLoader.build_optimization_dataset_path(dataset, base_name + "_majority_vote"), orient="records", lines=True)
    unanimous_vote_performances_df.to_json(ConfigLoader.build_optimization_dataset_path(dataset,base_name + "_unanimous_vote"), orient="records", lines=True)
    percentage_treshold_performces_df.to_json(ConfigLoader.build_optimization_dataset_path(dataset, base_name + "_percentage_treshold"), orient="records", lines=True)
    percentage_treshold_0_5_performces_df.to_json(ConfigLoader.build_optimization_dataset_path(dataset, base_name + "_percentage_treshold_0_5"), orient="records", lines=True)
    auc_pr_df.to_json(ConfigLoader.build_optimization_dataset_path(dataset, base_name + "_auc_pr"), orient="records", lines=True)
    print("-----------------------------\n"*3)
    # find the best ensemble by f1 score
    print("F1 score")
    best_majority_vote_ensemble = majority_vote_performances_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_unanimous_vote_ensemble = unanimous_vote_performances_df.sort_values(by="f1_score", ascending=False).iloc[0]

    print("percentage treshold 0.5",percentage_treshold_0_5_performces_df)
    best_percentage_treshold_0_5_ensemble = percentage_treshold_0_5_performces_df.sort_values(by="f1_score", ascending=False).iloc[0]
    print(f"Best percentage treshold 0.5 ensemble: {best_percentage_treshold_0_5_ensemble['ensemble']} with f1 score: {best_percentage_treshold_0_5_ensemble['f1_score']}")
    print("-----------------------------")

    print(f"Best majority vote ensemble: {best_majority_vote_ensemble['ensemble']} {best_majority_vote_ensemble['prompt_group']} with f1 score: {best_majority_vote_ensemble['f1_score']}")
    print(f"Best unanimous vote ensemble: {best_unanimous_vote_ensemble['ensemble']} {best_unanimous_vote_ensemble['prompt_group']} with f1 score: {best_unanimous_vote_ensemble['f1_score']}")
    print("-----------------------------")

    # find the best ensemble by accuracy
    print("Accuracy")
    best_majority_vote_ensemble = majority_vote_performances_df.sort_values(by="accuracy", ascending=False).iloc[0]
    best_unanimous_vote_ensemble = unanimous_vote_performances_df.sort_values(by="accuracy", ascending=False).iloc[0]

    print(f"Best majority vote ensemble: {best_majority_vote_ensemble['ensemble']} {best_majority_vote_ensemble['prompt_group']} with accuracy: {best_majority_vote_ensemble['accuracy']}")
    print(f"Best unanimous vote ensemble: {best_unanimous_vote_ensemble['ensemble']} {best_unanimous_vote_ensemble['prompt_group']} with accuracy: {best_unanimous_vote_ensemble['accuracy']}")
    print("-----------------------------")




    # find the best ensemble by precision
    print("Precision")
    best_majority_vote_ensemble = majority_vote_performances_df.sort_values(by="precision", ascending=False).iloc[0]
    best_unanimous_vote_ensemble = unanimous_vote_performances_df.sort_values(by="precision", ascending=False).iloc[0]

    print(f"Best majority vote ensemble: {best_majority_vote_ensemble['ensemble']} {best_majority_vote_ensemble['prompt_group']} with precision: {best_majority_vote_ensemble['precision']}")
    print(f"Best unanimous vote ensemble: {best_unanimous_vote_ensemble['ensemble']} {best_unanimous_vote_ensemble['prompt_group']} with precision: {best_unanimous_vote_ensemble['precision']}")
    print("-----------------------------")

    # find the best ensemble by recall
    print("Recall")
    best_majority_vote_ensemble = majority_vote_performances_df.sort_values(by="recall", ascending=False).iloc[0]
    best_unanimous_vote_ensemble = unanimous_vote_performances_df.sort_values(by="recall", ascending=False).iloc[0]
    print(f"Best majority vote ensemble: {best_majority_vote_ensemble['ensemble']} {best_majority_vote_ensemble['prompt_group']} with recall: {best_majority_vote_ensemble['recall']}")
    print(f"Best unanimous vote ensemble: {best_unanimous_vote_ensemble['ensemble']} {best_unanimous_vote_ensemble['prompt_group']} with recall: {best_unanimous_vote_ensemble['recall']}")

    print("-----------------------------")





           


if __name__ == "__main__":
    main()
    



    