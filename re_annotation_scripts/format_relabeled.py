import json
import os
import pandas as pd


def process_json(file_path):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print("file_path:", file_path)

    labeled_spans = []
    belongs_to = []
    original_text = None

    # Extract the relevant feature structures
    for item in data['%FEATURE_STRUCTURES']:
        if item['%TYPE'] == 'custom.Span':
            labeled_spans.append(item)
        elif item['%TYPE'] == 'webanno.custom.AlternativeLabel':
            belongs_to.append(item)
        elif item['%TYPE'] == 'uima.cas.Sofa':
            if original_text is not None:
                print("Warning: More than one 'uima.cas.Sofa' found.")
            original_text = item

    if original_text is None:
        print("Error: No 'uima.cas.Sofa' found.")
        return

    text = original_text.get('sofaString', '')

    # Create span_dict with extracted text
    span_dict = {}
    for span in labeled_spans:
        print("span:", span)
        span_text = text[span['begin']:span['end']]
        span["span"] = span_text
        if not "label" in span:
            print("Error: No label in span", span_text)
            span["label"] = "unlabeled"
            
        span_dict[span['%ID']] = {
            'label': span['label'],
            'span': span_text, 
            'begin': span['begin'],
            'end': span['end']
        }

    # Resolve belongs_to relations
    for relation in belongs_to:
        governor_id = relation.get('@Governor')
        dependent_id = relation.get('@Dependent')

        if governor_id in span_dict and dependent_id in span_dict:
            relation['from'] = span_dict[governor_id]
            relation['to'] = span_dict[dependent_id]

    # Store the results in a dictionary
    result = {
        'labeled_spans': labeled_spans,
        'belongs_to': belongs_to,
        'original_text': original_text
    }


    return result

def assemble_dataset_row(id, label_data):
    print("creating row for id: ", id)  
    row = {
        "id": id,
        "label": {},
    }

    
    # group the labeled spans by their label
    labels = [span for span in label_data["labeled_spans"] if span["label"] == "label" or span["label"] == "label_part"]
    alternative_labels = [span for span in label_data["labeled_spans"] if span["label"] == "alternative_label"]
    alternative_label_parts = [span for span in label_data["labeled_spans"] if span["label"] == "alternative_label_part"]
    unanswerable = [span for span in label_data["labeled_spans"] if  "unanswerable" in span["label"]  ]
    erronous_question = [span for span in label_data["labeled_spans"] if span["label"] == "erronous_question" or span["label"] == "erroneous_question"]
    label_candidates = [span for span in label_data["labeled_spans"] if span["label"] == "label_candidate" ]
    label_candidate_parts = [span for span in label_data["labeled_spans"] if span["label"] == "label_candidate_part"]
    # get other labels and their text for agreement computation includes unanswerable and erroneous_question as this is needed for agreement to filter out things that are written as additional label info on the question
    other_labels = [
        {
            "text": span["span"],
            "begin": span["begin"],
            "end": span["end"],
            "label": span["label"]
        }
        for span in label_data["labeled_spans"]
        if span["label"] not in ["label", "alternative_label", "alternative_label_part", "label_candidate", "label_candidate_part", "likely answer"]
    ]
    row["other_labels"] = other_labels
    print(
        "Found", len(labels), "labels ; ", 
        len(alternative_labels), "alternative_labels ;", 
        len(alternative_label_parts), "alternative_label_parts ;", 
        len(unanswerable), "unanswerable ;", 
        len(erronous_question), "erronous_question ;", 
        len(label_candidates), "label_candidates ;", 
        len(label_candidate_parts), "label_candidate_parts ;",
        len(other_labels), "other_labels")
    
    if len(other_labels) > 0:
        print("Warning: other_labels found:", other_labels)

    for span in labels:
        row["label"][span["span"]] = {
            "label": span["span"],
            "alternative_labels": [],
            "type": "label"
        }
    
    for span in label_candidates:
            row["label"][span["span"]] = {
                "label": span["span"],
                "alternative_labels": [],
                "type": "label_candidate"   
            }

    for span in unanswerable:
            row["unanswerable"] = True
    if len(unanswerable) == 0:
        row["unanswerable"] = False
    for span in erronous_question:
            row["erronous_question"] = True
    if len(erronous_question) == 0:
        row["erronous_question"] = False

    

    
    for span in alternative_labels:
        # check if it has one or more belongs_to relations
       
            # check if the belongs_to relation is connected to the current span
            found_match = False
            if label_data["belongs_to"]:
                for relation in label_data["belongs_to"]:
                    if relation["from"]["span"] == span["span"] and relation["to"]["label"] == "label":
                        found_match = True
                        # add the alternative label to the label
                        row["label"][relation["to"]["span"]]["alternative_labels"].append(span["span"])    
            if not found_match:
                # check if there is only one label, if so add it to the label
                
                if len(labels) == 1:
                    row["label"][labels[0]["span"]]["alternative_labels"].append(span["span"])
                elif row["unanswerable"] == True or row["erronous_question"] == True:
                    row["label"][span["span"]] = {
                        "label": span["span"],
                        "alternative_labels": [],
                        "type": "alternative_label"
                    }
                    
                else:
                    print("Error: alternative_label without belongs_to relation and multiple labels:",span["label"] )
    label2alternatives = group_alternative_label_parts(spans=label_data["labeled_spans"], relations=label_data["belongs_to"])
    for label in row["label"]:
        if label in label2alternatives:
            groups =  label2alternatives[label]
            for group in groups:
                if len(group) == 0:
                    continue
                print("group:", group)
                print("alternative parts:", [part["span"] for part in group]    )
                row["label"][label]["alternative_labels"].append(", ".join([alternative_part["span"] for alternative_part in group]))
            print("alternative labels grouped for label:", label, "are:", row["label"][label]["alternative_labels"])
    
    for span in label_candidate_parts:
        related_parts = []
        for relation in label_data["belongs_to"]:
            if relation["from"]["span"] == span["span"] and relation["to"]["label"] == "label_candidate":
                    if not "alternative_labels" in row["label"]:
                        row["label"]["alternative_labels"] = []
                    related_parts.append(relation["to"])
        if not related_parts or len(related_parts) == 0:
            print("Error: label_candidate_part without belongs_to relation ")
        else:
            # order by begin
            related_parts.sort(key=lambda x: x["begin"])
            # join to one string with ","
            joined = ", ".join([part["span"] for part in related_parts])
            row["label"][joined] = {
                "label": joined,
                "alternative_labels": [],
                "type": "label_candidate_part"
            }
    
    return row


# Process the JSON file
def create_labeled_dataset(annotation_folder, annotation_user="nedaniel"):
    labeled_dataset = {}
    
    # Iterate through all folders in the annotation folder
    for foldername in os.listdir(annotation_folder):
        folderpath = os.path.join(annotation_folder, foldername)
        if os.path.isdir(folderpath):
            if isinstance(annotation_user, str):
                json_filepath = os.path.join(folderpath, annotation_user + ".json")
            elif isinstance(annotation_user, list):
                for user in annotation_user:
                    potential_filepath = os.path.join(folderpath, user + ".json")
                    if os.path.exists(potential_filepath):
                        json_filepath = potential_filepath
                        break
                
            else:
                # Find the JSON file in the folder
                json_files = [f for f in os.listdir(folderpath) if f.endswith('.json')]
                if len(json_files) != 1:
                    raise Exception(f"Error: Expected 1 JSON file in folder {foldername}, found {len(json_files)}")
                else:
                    json_filepath = os.path.join(folderpath, json_files[0])
            if os.path.exists(json_filepath):
                # Process the JSON file
                result = process_json(json_filepath)
                # Store the result in the labeled_dataset dictionary
                labeled_dataset[foldername] = result

    # Filter out items that don't contain labeled_spans or belongs_to relations
    filtered_dataset = {k: v for k, v in labeled_dataset.items() if v['labeled_spans'] or v['belongs_to']}

    # clean the spans
    for id, label_data in filtered_dataset.items():
        belongs_to = label_data['belongs_to']
        labeled_spans = label_data['labeled_spans']
        for relation in belongs_to:
            # if there is a . in the span with a " " in front of it, remove the " "
            if relation["from"]["span"].find(" .") != -1:
                relation["from"]["span"] = relation["from"]["span"].replace(" .", ".")
            
            if relation["to"]["span"].find(" .") != -1:
                relation["to"]["span"] = relation["to"]["span"].replace(" .", ".")
        
        for span in labeled_spans:
            # if there is a . in the span with a " " in front of it, remove the " "
            if span["span"].find(" .") != -1:
                span["span"] = span["span"].replace(" .", ".")
            


    # Assemble the dataset rows
    dataset_rows = []
    for unformatted_id, label_data in filtered_dataset.items():
        id = unformatted_id.split(".")[0]
        dataset_rows.append(assemble_dataset_row(id, label_data))

    

    

    return dataset_rows


def group_alternative_label_parts(spans, relations):
    # Initialize dictionary to hold labels and their associated alternative label parts
    label_to_alternatives = {}

    # Function to find all transitively linked parts
    def find_transitive_links(start_span, visited, span_to_relation_map):
        
        group = []
        stack = [start_span]
        while stack:
            current_span = stack.pop()
            if current_span not in visited:
                visited.add(current_span)
                group.append(current_span)
                for relation in span_to_relation_map.get(current_span, []):
                    if relation['to']['label'] == 'alternative_label_part' and relation['to']['span'] not in visited:
                        stack.append(relation['to']['span'])
                    elif relation['from']['label'] == 'alternative_label_part' and relation['from']['span'] not in visited :
                        stack.append(relation['from']['span'])
        return group

    # Create a map from span to its relations for quick lookup
    span_to_relation_map = {}
    for relation in relations:
        if relation['from']['span'] not in span_to_relation_map:
            span_to_relation_map[relation['from']['span']] = []
        if relation['to']['span'] not in span_to_relation_map:
            span_to_relation_map[relation['to']['span']] = []
        span_to_relation_map[relation['from']['span']].append(relation)
        span_to_relation_map[relation['to']['span']].append(relation)

    # Find the unique label span if there is only one
    label_spans = [span['span'] for span in spans if span['label'] == 'label']
    unique_label_span = label_spans[0] if len(label_spans) == 1 else None

    # Find all direct links from alternative label parts to labels
    for relation in relations:
        if relation['from']['label'] == 'alternative_label_part' and relation['to']['label'] == 'label':
            label_span = relation['to']['span']
            alt_label_part_span = relation['from']['span']
            
            if label_span not in label_to_alternatives:
                label_to_alternatives[label_span] = set()
            
            label_to_alternatives[label_span].add(alt_label_part_span)

    # If there is only one label span, add all alternative_label_parts to it
    if unique_label_span:
        
        label_to_alternatives[unique_label_span] = {
            span["span"] for span in spans if span['label'] == 'alternative_label_part'
        }
        print("alternative label parts for unique label span:", label_to_alternatives[unique_label_span])


    # Traverse all direct links and find their transitive connections
    for label, alt_label_parts in label_to_alternatives.items():
        all_linked_groups = []
        visited = set()
        for part in alt_label_parts:
            if part not in visited:
                group = find_transitive_links(part, visited, span_to_relation_map)
                print("group:", group)
                # Convert spans to span dictionaries and sort them
                
                
                group_dicts = sorted(
                    [span for span in spans if span['span'] in group and span["label"] == "alternative_label_part"],
                    key=lambda x: x['begin']
                )
                print("group_dicts:", group_dicts)
                all_linked_groups.append(group_dicts)

        # Ensure the result is always a nested array
        label_to_alternatives[label] = all_linked_groups if all_linked_groups else [[]]

    print("label_to_alternatives:", label_to_alternatives)

    return label_to_alternatives









# Example usage
if __name__ == "__main__":
    annotation_folder = r"C:\Users\Daniel Neu\Downloads\webanno17200083956948968324export_curated_documents\curation"
    #annotation_user = ["annemarie.friedrich", "jakob.prange"]
    #annotation_user = "nedaniel"
    annotation_user= None
    output_file = r"dataset/relabeled/human_annotated_test" +("_"  if annotation_user is not None else ""	) + ( "-".join(annotation_user) if isinstance(annotation_user, list) else (annotation_user if annotation_user is not None else "")) + ".json"
    labeled_dataset = create_labeled_dataset(annotation_folder, annotation_user=annotation_user)
    # save to file
    df = pd.DataFrame(labeled_dataset)
    df.to_json(output_file, orient="records", lines=False, indent=4)

    print(labeled_dataset)
