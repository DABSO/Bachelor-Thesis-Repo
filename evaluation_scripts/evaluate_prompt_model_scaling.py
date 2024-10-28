import pandas as pd
import matplotlib.pyplot as plt
from src.ConfigLoader import ConfigLoader
from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

def create_auc_pr_scatter_plot(df_path, plot_path):
    """
    Creates a scatter plot comparing the effects of adding more models versus adding more prompts
    on the auc-pr scores.

    Parameters:
    - df_path (str): Path to the CSV file containing the dataframe.
    - plot_path (str): Path where the plot image will be saved.
    """
    # Load the dataframe
    df = pd.read_json(df_path, orient="records", lines=True, dtype=False)

    # Split the comma-separated strings into lists
    df['prompt_list'] = df['prompt_group'].apply(lambda x: x.split(','))
    df['model_list'] = df['ensemble'].apply(lambda x: x.split(','))

    # Identify all unique prompts and models
    unique_prompts = set([prompt for sublist in df['prompt_list'] for prompt in sublist])
    unique_models = set([model for sublist in df['model_list'] for model in sublist])

    max_prompts = len(unique_prompts)
    max_models = len(unique_models)

    # Function to filter dataframe for models
    def filter_by_models(k):
        return df[df['model_list'].apply(len) == k]

    # Function to filter dataframe for prompts
    def filter_by_prompts(k):
        return df[df['prompt_list'].apply(len) == k]

    # Prepare data for models
    model_data = []
    for k in range(1, max_models + 1):
        filtered_df = filter_by_models(k)
        # Further filter to have maximum number of prompts
        filtered_df = filtered_df[filtered_df['prompt_list'].apply(len) == max_prompts]
        model_data.append({
            'k': k,
            'auc_pr': filtered_df['auc_pr_df'].values
        })

    print("number of datapoints for models ", len(model_data))

    # Prepare data for prompts
    prompt_data = []
    for k in range(1, max_prompts + 1):
        filtered_df = filter_by_prompts(k)
        # Further filter to have maximum number of models
        filtered_df = filtered_df[filtered_df['model_list'].apply(len) == max_models]
        prompt_data.append({
            'k': k,
            'auc_pr': filtered_df['auc_pr_df'].values
        })

    print("number of datapoints for prompts", len(prompt_data))

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot for models
    model_aucs = []
    for k in range(0, max_models + 1):
        aucs = []
        for data in model_data:
            if data['k'] == k:
                aucs = data['auc_pr']
                break
        if len(aucs) > 0:
            model_aucs.append((k, aucs.mean()))
            plt.scatter([k]*len(aucs), aucs, color='blue', alpha=0.6, label='Models' if k == 0 else "")
    
    # Plot average line for models
    model_avg_x = [item[0] for item in model_aucs]
    model_avg_y = [item[1] for item in model_aucs]
    plt.plot(model_avg_x, model_avg_y, color='blue', label='k Models w Main Prompts Average', linewidth=2.5)

    # Plot for prompts
    prompt_aucs = []
    for k in range(0, max_prompts + 1):
        aucs = []
        for data in prompt_data:
            if data['k'] == k:
                aucs = data['auc_pr']
                break
        if len(aucs) > 0:
            prompt_aucs.append((k, aucs.mean()))
            plt.scatter([k]*len(aucs), aucs, color='orange', alpha=0.6, label='Prompts' if k == 0 else "")
    
    # Plot average line for prompts
    prompt_avg_x = [item[0] for item in prompt_aucs]
    prompt_avg_y = [item[1] for item in prompt_aucs]
    plt.plot(prompt_avg_x, prompt_avg_y, color='orange', label='k Prompts w Large Models Average', linewidth=2.5)
    # Increase font size and line width
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['lines.linewidth'] = 2.5
    
    # Labels and title
    plt.xlabel('Number of Models / Prompts')
    plt.xticks(list(range(1, max(max_models, max_prompts) + 1)))
   
    
    plt.xlabel('Number of Models / Prompts', fontsize=20)
    plt.ylabel('AUC-PR Score', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

# Example usage
if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--prompt_groups", required=True)
    parser.add_argument("--model_groups", required=True)

    

    args = parser.parse_args()
    dataset = args.dataset
    base_name =  args.prompt_groups.replace(",", "-") + "_" + args.model_groups.replace(",", "-")

    data_path = ConfigLoader.build_optimization_dataset_path(dataset, base_name + "_    auc_pr")
    plot_path = "evaluations/auc_pr_scaling_"+dataset
    create_auc_pr_scatter_plot(data_path, plot_path)
