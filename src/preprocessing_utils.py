import re
import pandas as pd

def reduce_context(row):
    labels = row['label'].split(';')
    context = row['context']

    # remove empty labels
    labels = [label for label in labels if label.strip()]
    
    # Function to generate valid substrings
    def generate_valid_substrings(label):
        min_substring_length = max(len(label) // 3, len(label.split()[0])) if label.split() else len(label) // 3
        return [label[i:j+1] for i in range(len(label)) for j in range(i, len(label)) 
                if len(label[i:j+1]) >= min_substring_length and ' ' in label[i:j+1]]
    
    # Combine all labels and their valid substrings
    substrings_to_search = []
    for label in labels:
        substrings_to_search.extend([label] + generate_valid_substrings(label.strip()))
    
    # Split context into sentences
    sentences = re.split(r'(?<=[.!?]) +', context)
    
    # Find all matches of the substrings in the context
    matched_indices = set()
    for substring in substrings_to_search:
        for match in re.finditer(re.escape(substring), context):
            match_start = match.start()
            match_end = match.end()
            
            # Find the sentences containing the match
            for i, sentence in enumerate(sentences):
                sentence_start = context.find(sentence)
                sentence_end = sentence_start + len(sentence)
                if sentence_start <= match_start < sentence_end or sentence_start < match_end <= sentence_end:
                    matched_indices.add(i)
    
    if not matched_indices:
        # If no match is found, return the row unchanged
        return row
    
    # always get the first 10 sentences 
    for i in range(10):
        if i < len(sentences):
            matched_indices.add(i)

    
    # Get 5 sentences before and after each matched sentence
    expanded_indices = set()
    for index in matched_indices:
        start_index = max(0, index - 5)
        end_index = min(len(sentences), index + 6)
        expanded_indices.update(range(start_index, end_index))
    
    # Sort the indices and add ellipses for gaps
    sorted_indices = sorted(expanded_indices)
    # Select the sentences
    selected_sentences = []
    last_index = 0  # Initialize to a value that ensures no ellipsis at the start
    for i in sorted_indices:
        if i > last_index + 1:
            selected_sentences.append("[...]")
        selected_sentences.append(sentences[i].strip())
        last_index = i
    
    # Join selected sentences to form the reduced context
    reduced_context = ' '.join(selected_sentences)
    
    # Join selected sentences to form the reduced context
    reduced_context = ' '.join(selected_sentences)

    print("Reduced context length from {} to {}".format(len(context), len(reduced_context)), "Reduction by ", (len(context) - len(reduced_context))/len(context) * 100, "%")
    print("Reduced Sentences from {} to {}".format(len(sentences), len(selected_sentences)))
    
    
    # Assign the reduced context back to the row
    row['context'] = reduced_context
    return row


def replace_empty_label(row):
    replacement = "Not specified in the given Context"
    if row["label"] == "":
        row["label"] = replacement

    return row

