from src.ConfigLoader import ConfigLoader
from argparse import ArgumentParser

from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--confidence_threshold", type=float, required=False)
parser.add_argument("--aggregation_column", type=str, required=False)
parser.add_argument("--preferred_model", type=str, required=True)
parser.add_argument("--filter", action="store_true", required=False)
parser.add_argument("--prompt_name", type=str)
parser.add_argument("--cot", action="store_true", required=False)
args = parser.parse_args()

id_column = ConfigLoader.get_dataset_config(args.dataset)["id_column"]
fine_tuning_dataset = pd.DataFrame(columns=[id_column, "input", "output"])


# load the predict answer prompt template
predict_answer_prompt_path = os.getenv("PROMPT_DIR") + "/" + ConfigLoader.get_prompt_config("predict_answer")["prompt_files"][-1]["file"]
predict_answer_prompt = ""
with open(predict_answer_prompt_path, "r") as file:
    predict_answer_prompt = file.read()

# format the input of each row
fine_tuning_dataset["input"] = fine_tuning_dataset["input"].apply(lambda x: predict_answer_prompt.format(context=x["context"], question=x["question"]))


if args.filter:
    aggregated_predictions_dataset = pd.read_json(ConfigLoader.get_aggregated_predictions_dataset_path(dataset_name=args.dataset), orient="records", lines=True, dtype=False)
    input_dataset = pd.read_json(ConfigLoader.get_input_dataset_path(dataset_name=args.dataset), orient="records", lines=True, dtype=False)
    # get the ids of the confident and correct predictions
    confident_correct_instance_ids = aggregated_predictions_dataset[aggregated_predictions_dataset[args.aggregation_column] > args.confidence_threshold][id_column].tolist()
    print("confident_correct_instance_ids: ", len(confident_correct_instance_ids))
    # filter the input dataset
    confident_correct_instances = input_dataset[input_dataset[id_column].isin(confident_correct_instance_ids)]

    # Ensure we have the filtered input dataset
    confident_correct_instances = confident_correct_instances.set_index(id_column)

    # get the predictions of the models
    predictions_dataset = pd.read_json(ConfigLoader.get_prediction_dataset_path(dataset_name=args.dataset, prompt_name="predict_answer"), orient="records", lines=True, dtype=False)
    # get the rows of the predictions dataset that are in the confident_correct_predictions_dataset
    confident_correct_predictions_predictions_dataset = predictions_dataset[predictions_dataset[id_column].isin(confident_correct_instance_ids)]
    # get available models
    available_models = ConfigLoader.load_available_models()
    # get the rows of the predictions dataset that are in the available 
    preferred_model_column = ConfigLoader.get_model_config(args.preferred_model)["output_column"]
    model_output_columns = [ConfigLoader.get_model_config(model)["output_column"] for model in available_models if model != args.preferred_model]
    all_model_output_columns = [preferred_model_column] + model_output_columns 
    
    # get the correct outputs of the models and create the fine-tuning dataset
    if args.cot:

        id2_correct_model_predictions = {}
        for i, row in confident_correct_predictions_predictions_dataset.iterrows():
            id2_correct_model_predictions[row[id_column]] = None

        for model_output_column in all_model_output_columns:
            if model_output_column in row and row[model_output_column] == True:
                print("model_output_column: ", model_output_column)
                id2_correct_model_predictions[row[id_column]] = model_output_column
                break
        # remove none values	
        id2_correct_model_predictions = {k: v for k, v in id2_correct_model_predictions.items() if v is not None}
        print("remaining instances: ", len(id2_correct_model_predictions))

        # get the outputs of the models
        model_output_dataset = pd.read_json(ConfigLoader.get_model_output_dataset_path(dataset_name=args.dataset, prompt_name="predict_answer"), orient="records", lines=True, dtype=False)
        
        for i, row in model_output_dataset.iterrows():
            if row[id_column] in id2_correct_model_predictions:
                correct_model = id2_correct_model_predictions[row[id_column]]
                
                # Get the corresponding input data
                input_row = confident_correct_instances.loc[row[id_column]]
                
                # Create a new row for the fine-tuning dataset
                fine_tuning_row = {
                    id_column: row[id_column],
                    "input": predict_answer_prompt.format(context=input_row["context"], question=input_row["question"]),

                    "output": row[correct_model]
                }
                fine_tuning_dataset = fine_tuning_dataset._append(fine_tuning_row, ignore_index=True)
    else:
        # simply format the input of each row and set original_answer as output
        #TODO: add the case where we use filter without CoT
        pass


else:
    # TODO: add the case where we don't filter
    pass



# save the dataset
fine_tuning_dataset_name = args.dataset + "_" + args.aggregation_column + "_" + str(args.confidence_threshold) 
fine_tuning_dataset.to_json(ConfigLoader.get_fine_tuning_dataset_path(dataset_name=fine_tuning_dataset_name, prompt_name="predict_answer"), orient="records", lines=True)
