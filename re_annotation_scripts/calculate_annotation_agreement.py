from src.ConfigLoader import ConfigLoader
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

ANNOTATOR_1 =  "nedaniel"
ANNOTATOR_2 = "annemarie.friedrich-jakob.prange" 
dataset_name = "human_annotated_test"
ann1_dataset = pd.read_json(ConfigLoader.get_relabeled_dataset_path(dataset_name + ( "_" + ANNOTATOR_1 if ANNOTATOR_1 else "")), orient="records", dtype=False)
ann2_dataset = pd.read_json(ConfigLoader.get_relabeled_dataset_path(dataset_name + ( "_" + ANNOTATOR_2 if ANNOTATOR_2 else "")), orient="records", dtype=False)
id_column = ConfigLoader.get_dataset_config(dataset_name)["id_column"]
# get only the rows that are in both datasets by id
ann1_dataset = ann1_dataset[ann1_dataset[id_column].isin(ann2_dataset[id_column])]
ann2_dataset = ann2_dataset[ann2_dataset[id_column].isin(ann1_dataset[id_column])]

print(ann1_dataset.head())
print(ann2_dataset.head())
def normalize_label(label):
        return label.strip().lower()
def get_annotations_flat(row):
    # get all labels, alternative labels, and other labels as a single list of strings
    labels = list(row["label"].keys())
    labels = [normalize_label(label) for label in labels]
    alternative_labels = [v["alternative_labels"] for v in row["label"].values()]
    alternative_labels = [item for sublist in alternative_labels for item in sublist]
    alternative_labels = [normalize_label(label) for label in alternative_labels]
    if "other_labels" in row:
        other_labels = [v["text"] for v in row["other_labels"]]
    else:
        other_labels = []
    other_labels = [normalize_label(label) for label in other_labels]

    return set(labels + alternative_labels + other_labels)


def agreement_unanswerable(ann1_dataset, ann2_dataset):
    # Ensure both datasets have the same index
    ann1_dataset = ann1_dataset.set_index(id_column)
    ann2_dataset = ann2_dataset.set_index(id_column)
    
    # Sort both datasets by index to ensure alignment
    ann1_dataset = ann1_dataset.sort_index()
    ann2_dataset = ann2_dataset.sort_index()
    
    # Compare 'unanswerable' columns
    agreement = (ann1_dataset['unanswerable'] == ann2_dataset['unanswerable'])
    
    return agreement.mean()

def intersection_over_union_row(ann1_row, ann2_row):
    ann1_labels = get_annotations_flat(ann1_row)
    ann2_labels = get_annotations_flat(ann2_row)
    return len(ann1_labels.intersection(ann2_labels)) / len(ann1_labels.union(ann2_labels))

def intersection_over_union(ann1_dataset, ann2_dataset):

    def process_row(row):
        try:
            ann2_row = ann2_dataset[ann2_dataset[id_column] == row[id_column]].iloc[0]
            return intersection_over_union_row(row, ann2_row)
        except Exception as e:
            print(f"Error processing row with id {row[id_column]}: {e}")
            return None
    
    results = ann1_dataset.apply(process_row, axis=1)
    
    # Remove None values before calculating mean
    valid_results = results.dropna()
    
    return valid_results.mean()

def intersection_over_union_with_substrings(ann1_dataset, ann2_dataset, filter_unanswerable=True):

    # filter those out where at least one is unanswerable
    if filter_unanswerable:
        # Filter out rows where either dataset has "unanswerable" set to True
        unanswerable_ids = set(ann1_dataset[ann1_dataset["unanswerable"] == True][id_column]) | set(ann2_dataset[ann2_dataset["unanswerable"] == True][id_column])
        ann1_dataset = ann1_dataset[~ann1_dataset[id_column].isin(unanswerable_ids)]
        ann2_dataset = ann2_dataset[~ann2_dataset[id_column].isin(unanswerable_ids)]

    ious = []
    
    for index, row in ann1_dataset.iterrows():
        ann1_labels = get_annotations_flat(row)
        ann2_labels = get_annotations_flat(ann2_dataset[ann2_dataset[id_column] == row[id_column]].iloc[0])
            
        intersections = 0
        union = len(set(ann1_labels).union(set(ann2_labels)))

        for ann1_label in ann1_labels:
            for ann2_label in ann2_labels:
                if ann1_label == ann2_label:
                    intersections += 1
                elif ann1_label in ann2_label:
                    intersections += 1
                elif ann2_label in ann1_label:
                    intersections += 1

        ious.append(intersections / union)
    
    return sum(ious) / len(ious)





import pandas as pd
from typing import List, Dict, Any

def calculate_entity_agreement_iou(df1: pd.DataFrame, df2: pd.DataFrame, skip_unanswerable: bool = True, duty_only: bool = False, verbose: bool = False) -> float:
    def extract_entity_options(row: Dict[str, Any]) -> List[Dict[str, List[str]]]:
        entity_options = []
        if isinstance(row.get('label'), dict):
            for entity_info in row['label'].values():
                if duty_only and entity_info["type"] != "label":
                    continue
                options = {
                    'main': entity_info['label'].lower(),
                    'alternatives': [alt.lower() for alt in entity_info.get('alternative_labels', [])]
                }
                entity_options.append(options)
        return entity_options

    def entities_match(entity_a: Dict[str, List[str]], entity_b: Dict[str, List[str]]) -> bool:
        all_options_a = [entity_a['main']] + entity_a['alternatives']
        all_options_b = [entity_b['main']] + entity_b['alternatives']
        return any(a in b or b in a or a.lower() in b.lower() or b.lower() in a.lower() for a in all_options_a for b in all_options_b)

    def calculate_row_iou(row1: Dict[str, Any], row2: Dict[str, Any]) -> float:
        entities1 = extract_entity_options(row1)
        entities2 = extract_entity_options(row2)

        intersection = 0
        matched_entities1 = set()
        matched_entities2 = set()

        for i, e1 in enumerate(entities1):
            for j, e2 in enumerate(entities2):
                if j not in matched_entities2 and entities_match(e1, e2):
                    intersection += 1
                    matched_entities1.add(i)
                    matched_entities2.add(j)
                    break

        union = len(entities1) + len(entities2) - intersection

        if verbose:
            unmatched_entities1 = set(range(len(entities1))) - matched_entities1
            unmatched_entities2 = set(range(len(entities2))) - matched_entities2
            if unmatched_entities1 or unmatched_entities2:
                print(f"row: {row1[id_column]}")
                print(f"entities in 1 that are not in 2: {[entities1[i] for i in unmatched_entities1]}")
                print(f"entities in 2 that are not in 1: {[entities2[i] for i in unmatched_entities2]}")
                print()

        return intersection / union if union > 0 else 0

    iou_scores = []

    for _, row1 in df1.iterrows():
        row2 = df2.loc[df2[id_column] == row1[id_column]].iloc[0]

        if skip_unanswerable and (row1.get('unanswerable', False) or row2.get('unanswerable', False)):
            continue

        iou_scores.append(calculate_row_iou(row1, row2))

    return sum(iou_scores) / len(iou_scores) if iou_scores else 0



                        
if __name__ == "__main__":
    print( "agreement_unanswerable: ", agreement_unanswerable(ann1_dataset, ann2_dataset))
    print( "intersection_over_union: ", intersection_over_union(ann1_dataset, ann2_dataset))
    print( "intersection_over_union_with_substrings: ", intersection_over_union_with_substrings(ann1_dataset, ann2_dataset))	
    print( "entity_intersection_over_union: ",calculate_entity_agreement_iou(ann1_dataset, ann2_dataset, verbose=True))


