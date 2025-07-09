import pandas as pd
import os
import re
from typing import List, Tuple
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import numpy as np

# --- Configuration ---
data_path = '/Users/pankajti/dev/data/kaggle/make-data-count-finding-data-references'
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the label mapping for NER
# O = 0, B-DATASET_ID = 1, I-DATASET_ID = 2
ner_label_map = {"O": 0, "B-DATASET_ID": 1, "I-DATASET_ID": 2}
id_to_label_ner = {v: k for k, v in ner_label_map.items()}

# Define the label mapping for Classification
# You'll need to define this based on your 'type' column values
classification_label_map = {"primary": 0, "secondary": 1,'missing':2}
id_to_label_classification = {v: k for k, v in classification_label_map.items()}


# --- Helper Function (Modified for Subword Tokenization) ---
def create_bio_labels_for_sentence(sentence: str, target_dataset_id: str, tokenizer) -> Tuple[List[int], List[int]]:
    """
    Tokenizes a sentence and creates BIO labels for a given target_dataset_id mention.
    Handles subword tokenization by ensuring only the first subword gets 'B-' and subsequent 'I-'.

    Args:
        sentence (str): The input sentence.
        target_dataset_id (str): The dataset_id string to look for (e.g., "10.5281/zenodo.1234567").
        tokenizer: The pre-trained model's tokenizer.

    Returns:
        Tuple[List[int], List[int]]: A tuple of token IDs and their corresponding BIO labels.
    """
    tokenized_input = tokenizer(
        sentence,
        return_offsets_mapping=True, # Important for mapping back to original text
        add_special_tokens=True,    # Add CLS/SEP tokens
        truncation=True,            # Truncate if too long
        max_length=512              # Max length for SciBERT
    )
    # offset_mapping gives (start_char, end_char) for each token
    offsets = tokenized_input['offset_mapping']
    input_ids = tokenized_input['input_ids']

    labels = [ner_label_map["O"]] * len(input_ids)

    if pd.isna(target_dataset_id): # If no ground truth dataset_id, all 'O'
        return input_ids, labels

    # --- Robustly find the dataset_id span in the original sentence ---
    # This requires using the exact logic from your `is_dataset_id_in_sentence`
    # but modified to return the *span* if a match is found.
    # For simplicity, let's assume direct string finding for now.
    # In a real scenario, you'd integrate/refine your `is_dataset_id_in_sentence`
    # to return (start_char, end_char) of the matched ID.

    match_span = None

    # First, try to find the exact target_dataset_id
    escaped_id = re.escape(target_dataset_id)
    # Use word boundaries if possible to prevent partial matches like "1.2.3" in "1.2.345"
    match = re.search(r'\b' + escaped_id + r'\b', sentence, re.IGNORECASE)
    if match:
        match_span = match.span()
    else:
        # Fallback to DOI pattern if exact match not found, similar to your original logic
        doi_pattern = re.compile(r'10\.\d{4,9}/\S+', re.I)
        doi_matches = doi_pattern.finditer(sentence)
        for doi_m in doi_matches:
            # Check if the ground truth dataset_id is part of the detected DOI
            if target_dataset_id.lower() in doi_m.group(0).lower():
                match_span = doi_m.span()
                break
    # Add other common patterns for dataset_ids if they exist (e.g., accessions)
    # Example: PDB IDs, accession numbers like "GSE12345"
    # if not match_span:
    #     pdb_match = re.search(r'\b[1-9][a-zA-Z0-9]{3}\b', sentence)
    #     if pdb_match and target_dataset_id.lower() == pdb_match.group(0).lower():
    #         match_span = pdb_match.span()


    if match_span:
        mention_start_char, mention_end_char = match_span

        previous_offset_end = 0 # To track if a token starts immediately after a special token

        for i, (token_start_char, token_end_char) in enumerate(offsets):
            # Skip special tokens ([CLS], [SEP]) and tokens outside the mention
            if token_start_char == token_end_char: # Special tokens have (0,0) offset mapping
                labels[i] = -100 # Ignore during loss calculation
                continue

            # Check if the token's span overlaps with the mention's span
            if token_end_char > mention_start_char and token_start_char < mention_end_char:
                # If the current token's start char is within the mention:
                if token_start_char >= mention_start_char:
                    # Check if this is the start of a new entity token
                    # This handles subword tokens correctly (B- only for first piece)
                    if i > 0 and (offsets[i-1][1] <= token_start_char or labels[i-1] == ner_label_map["O"]): # If previous token ended before this, or was 'O'
                         labels[i] = ner_label_map["B-DATASET_ID"]
                    elif i==0 and token_start_char == mention_start_char: # First token is B
                        labels[i] = ner_label_map["B-DATASET_ID"]
                    elif labels[i-1] in [ner_label_map["B-DATASET_ID"], ner_label_map["I-DATASET_ID"]]: # Continuation of entity
                        labels[i] = ner_label_map["I-DATASET_ID"]
                    else: # Fallback to O if logic is tricky, prevents accidental labeling
                         labels[i] = ner_label_map["O"]
                else: # Token starts before mention but overlaps, likely a subword
                    labels[i] = ner_label_map["I-DATASET_ID"]
            else:
                labels[i] = ner_label_map["O"]
    # If no match_span, all labels remain `ner_label_map["O"]` (or -100 for special tokens)

    # Ensure special tokens are -100
    labels[0] = -100 # [CLS]
    if len(labels) > 0:
        labels[-1] = -100 # [SEP]

    return input_ids, labels


# --- Load Merged DataFrame ---
merged_df = pd.read_csv("../data/merged_df.csv") # Assuming you saved it here

# Remove duplicate rows based on (sentence, dataset_id, article_id, type)
# Keeping only the first occurrence for unique (sentence, dataset_id, primary_secondary) pairs
merged_df = merged_df.drop_duplicates(subset=['sentence', 'dataset_id', 'article_id', 'type'])


# --- Prepare NER Training Data ---
ner_data = []
# Filter to relevant rows for NER training
# We want to train on:
# 1. 'both' (LLM flagged + GT matched) - definite positives
# 2. 'right_only' (GT missed by LLM/rules) - missed positives, need to search original text
# 3. 'left_only' (LLM flagged, but not in GT) - can be used as negative examples (all 'O') or filtered out
#    For simplicity, let's include 'left_only' as sentences where no GT dataset_id is the target.
#    This means `target_dataset_id` will be NaN for these, leading to all 'O' labels.

# It's crucial to go back to original XML for 'right_only' if the sentence is not in merged_df
# For the sake of this example, let's simplify and assume all 'right_only' have sentences
# in merged_df, which isn't always true for the FN case you identified.
# For a full solution, you'd need to re-extract sentences for 'right_only' article_ids.
# For now, let's work with the sentences in `merged_df`.

ner_train_df = merged_df[merged_df['_merge'] != 'left_only'] # Focus on known GT and 'both'
ner_train_df = ner_train_df.reset_index(drop=True)

# Generate NER features and labels
for idx, row in ner_train_df.iterrows():
    sentence = str(row['sentence']) # Ensure it's a string
    # Use the ground truth dataset_id if available, otherwise None for 'O' labels
    target_id = row['dataset_id'] if row['_merge'] == 'both' or row['_merge'] == 'right_only' else None

    # Handle cases where `dataset_id` might be NaN in the merged_df from `detected_df`
    # and not present in `train_labels` (e.g., pure `left_only` or LLM false positive)
    if pd.isna(target_id):
        # Treat as a negative example for NER (all 'O' labels)
        input_ids, labels = create_bio_labels_for_sentence(sentence, None, tokenizer)
    else:
        input_ids, labels = create_bio_labels_for_sentence(sentence, target_id, tokenizer)

    ner_data.append({
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels
    })

# Convert to Hugging Face Dataset
ner_hf_dataset = Dataset.from_list(ner_data)

# Print a sample to check
print("\n--- Sample NER Data ---")
sample_ner_idx = 0
print(f"Sentence: {tokenizer.decode(ner_hf_dataset[sample_ner_idx]['input_ids'])}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(ner_hf_dataset[sample_ner_idx]['input_ids'])}")
print(f"Labels: {[id_to_label_ner[l if l != -100 else 0] for l in ner_hf_dataset[sample_ner_idx]['labels']]}")


# --- Prepare Classification Training Data ---
# For classification, we only care about sentences where we have a ground truth `primary_secondary` label.
classification_data = []
classification_train_df = merged_df[merged_df['_merge'] != 'left_only'].copy() # Focus on 'both' and 'right_only'
classification_train_df = classification_train_df.drop_duplicates(subset=['sentence', 'article_id', 'type']) # One label per (sentence, type)
classification_train_df = classification_train_df.reset_index(drop=True)

for idx, row in classification_train_df.iterrows():
    sentence = str(row['sentence'])
    # Only use rows where 'type' (primary_secondary) is available
    if pd.notna(row['type']):
        classification_data.append({
            'text': sentence,
            'label': classification_label_map[row['type'].lower()]
        })

# Convert to Hugging Face Dataset
classification_hf_dataset = Dataset.from_list(classification_data)

# Tokenize for classification model
def tokenize_function_clf(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_classification_hf_dataset = classification_hf_dataset.map(tokenize_function_clf, batched=True)

# Print a sample to check
print("\n--- Sample Classification Data ---")
sample_clf_idx = 0
print(f"Text: {tokenized_classification_hf_dataset[sample_clf_idx]['text']}")
print(f"Label: {id_to_label_classification[tokenized_classification_hf_dataset[sample_clf_idx]['label']]}")

# --- Train/Validation Split ---
# It's crucial to split your datasets into training and validation sets.
ner_train_val_split = ner_hf_dataset.train_test_split(test_size=0.1, seed=42)
ner_train_dataset = ner_train_val_split['train']
ner_val_dataset = ner_train_val_split['test']

clf_train_val_split = tokenized_classification_hf_dataset.train_test_split(test_size=0.1, seed=42)
clf_train_dataset = clf_train_val_split['train']
clf_val_dataset = clf_train_val_split['test']

print(f"\nNER Train size: {len(ner_train_dataset)}, NER Val size: {len(ner_val_dataset)}")
print(f"Classification Train size: {len(clf_train_dataset)}, Classification Val size: {len(clf_val_dataset)}")



