import pandas as pd 

INPUT_FILE_PATH = r"dataset/original/human_annotated_test.json"
OUTPUT_FILE_PATH = r"dataset/input/human_annotated_test.json"

def reformat_human_annotated_file(input_filepath, output_filepath):
        # Load data
    df = pd.read_json(input_filepath, orient='records')

    # Identify the original columns and the ones to be handled explicitly
    all_columns = set(df.columns)
    handled_columns = {'paras', 'questions', 'index'}
    other_columns = list(all_columns - handled_columns)

    # Prepare the new DataFrame with additional columns
    column_names = ['context', 'question', 'label', 'position_labels', 'index', 'id'] + other_columns

    formatted_data = []
    for i in range(len(df)):
        row_data = df.iloc[i]
        paras = row_data['paras']
        index = row_data['index']
        context = "\n".join(paras)
        qa_pairs = row_data['questions']
        
        for j, qa in enumerate(qa_pairs):
            question = qa[0]
            labels = qa[1]
            label_text = " ; ".join(label["answer"] for label in labels)
            label_text = label_text.replace(" .", ".")
            try:
                position_labels = [{"from": label["from"], "end": label["end"], "para": label["para"], "target": labels[i]} for i, label in enumerate(labels)]
            except KeyError:
                position_labels = None
                print(labels)

            new_row = {
                "id": str(i) + "_" + str(j),
                "index": index,
                'context': context,
                'question': question,
                'label': label_text,
                'position_labels': position_labels
            }

            for col in other_columns:
                new_row[col] = row_data[col]
            
            formatted_data.append(new_row)
        
    result_df = pd.DataFrame(formatted_data, columns=column_names)

    result_df.to_json(output_filepath, orient='records', lines=True)





if __name__ == "__main__":
    reformat_human_annotated_file(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    print("Done!")



