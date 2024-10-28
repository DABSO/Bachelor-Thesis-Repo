import pandas as pd 
from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv
import json
import argparse
load_dotenv()



# parse args 
available_datasets = ConfigLoader.load_available_datasets()

argparser = argparse.ArgumentParser()	
argparser.add_argument("--dataset", type=str, required=True, choices=available_datasets)
argparser.add_argument("--redo", action="store_true", required=False, help="Redo the evaluation of the whole dataset")
args = argparser.parse_args()
dataset = args.dataset
redo = args.redo or False
id_column = ConfigLoader.get_dataset_config(dataset)["id_column"]
print("redo:", redo)

original_dataset_path = ConfigLoader.get_input_dataset_path(dataset)

relabeled_dataset_path = ConfigLoader.get_relabeled_dataset_path(dataset)

original_df = pd.read_json(original_dataset_path, orient='records', lines=True,dtype=False)

relabeled_df = pd.read_json(relabeled_dataset_path, orient='records', lines=False,dtype=False) # TODO: adjust this to read line by line

# sort examples by id 
original_df = original_df.sort_values(id_column)
relabeled_df = relabeled_df.sort_values(id_column)

print(len(original_df))
print(len(relabeled_df))


# check if relabeled labels match the original data
def label_matches_relabeled(relabeled_row, original_row):
    print("original_label", original_row["label"])
    print("relabeled", relabeled_row["label"])

    if original_row["label"] == "" and relabeled_row["unanswerable"] is True:
        print("both are unanswerable", original_row["label"], relabeled_row["unanswerable"])
        return True
    elif original_row["label"] != "" and relabeled_row["unanswerable"] is True and relabeled_row["label"] == {}:
        print("relabeled is unanswerable but original says it is not", original_row["label"], relabeled_row["label"])
        return False

    original_labels = [l.strip() for l in original_row["label"].split(";")]
    # check if all labels or their alternatives exist in the original row 
    
    for k, v in relabeled_row["label"].items():
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

        if not found_match and v["type"] != "label_candidate": # there is a label in the relabeled data that does not match the original
            
            print("no match found", False)
            return False
    if original_labels: # there are labels in the original data that are not in the relabeled data
        print("some labels are not in the relabeled data", original_labels)
        return False
    
    print("all labels match", True)
    return True

matching_labels_count = 0

evaluated_relabeled_df = pd.DataFrame(columns=[*relabeled_df.columns, "original_label", "matches_original", "manual_review"])
existing_evaluated_relabeled_df = pd.read_json(ConfigLoader.get_relabeled_evaluation_dataset_path(dataset), orient='records', lines=True, dtype=False)


 


# Populate evaluated_relabeled_df with matches
for i, row in relabeled_df.iterrows():
    original_row = original_df[original_df[id_column] == row[id_column]]
    row["original_label"] = original_row.iloc[0]["label"]
    row["matches_original"] = label_matches_relabeled(row, original_row.iloc[0])
    if row["matches_original"]:
        matching_labels_count += 1
    row["question"] = original_row.iloc[0]["question"]
    evaluated_relabeled_df = evaluated_relabeled_df._append(row, ignore_index=True)

#Manual Review
review_stack = []

i = 0  # Start at the beginning of the dataframe
while i < len(evaluated_relabeled_df):
    row = evaluated_relabeled_df.iloc[i]
    print("checking if ", row[id_column], "is in", existing_evaluated_relabeled_df[id_column].values)
    if not args.redo and row[id_column] in existing_evaluated_relabeled_df[id_column].values: # skip existing evaluations unless --redo is set
        # set the row to the existing evaluation
        evaluated_relabeled_df.loc[i] = existing_evaluated_relabeled_df.loc[existing_evaluated_relabeled_df[id_column] == row[id_column]].iloc[0]
        i += 1
        continue
    
    if not row["matches_original"] and pd.isna(row.get("manual_review")):
        print("--------------------")
        print("ID:", row[id_column])
        print("Question:", row["question"])
        print("Original Label:", row["original_label"])
        print("Relabeled Label:")
        print(json.dumps(row["label"], indent=4))
        print("-------")
        print("Current index:", i)
        
        review_result = input("Do the labels match? (y/n) or go back (b): ").lower()
        
        if review_result == "b" and review_stack:
            # Going back: pop from the stack and undo changes
            last_index, last_result = review_stack.pop()
            evaluated_relabeled_df.loc[last_index, "matches_original"] = None
            evaluated_relabeled_df.loc[last_index, "manual_review"] = None
            if last_result == "y":
                matching_labels_count -= 1
            print(f"Going back to index {last_index}...")
            i = last_index  # Move back to the previous index
            continue
        
        elif review_result in ("y", "n"):
            evaluated_relabeled_df.loc[i, "matches_original"] = (review_result == "y")
            evaluated_relabeled_df.loc[i, "manual_review"] = True
            review_stack.append((i, review_result))  # Save index and result for potential "go back"
            if review_result == "y":
                matching_labels_count += 1
        
        else:
            print("Invalid input. Please enter 'y', 'n', or 'b' to go back.")
            continue  # Skip to the next iteration without advancing
        
        print("--------------------")
    
    i += 1  # Move to the next row if no "go back"


# save to file 
print(f"Percentage of matching labels: {matching_labels_count/len(relabeled_df)}")
evaluated_relabeled_df.to_json(ConfigLoader.get_relabeled_evaluation_dataset_path((args.dataset)), orient='records', lines=True)


