import json
import os
INPUT_FILE = r'dataset/original/human_annotated_train.json'
OUTPUT_DIR = r'dataset/annotation_docs/human_annotated_train'  

# Load the JSON file
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

# Iterate over each row in the data
for i, row in enumerate(data):
    # Join the context paragraphs with "\n"
    context = "\n".join(row['paras'])

    # Iterate over each question
    for j, question in enumerate(row['questions']):
        # Create a new file for each question
        filename = f'{OUTPUT_DIR}/{i}_{j}.txt'
        with open(filename, 'w', encoding="utf-8") as f:
            # Write the context and question to the file
            f.write(context + "\n" + 50*"=" + "\n\n\n." + question[0],)

