import pandas as pd
import os

data_path = '/Users/pankajti/dev/data/kaggle/make-data-count-finding-data-references'

# Load all LLM-detected sentences across first 10 articles
detected_df = pd.read_csv(os.path.join("/Users/pankajti/dev/git/random_research/random_research/20250704","all_detected_sentences.csv"))

# Load train labels
train_labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))

merged_df = detected_df.merge(train_labels, how='outer',indicator=True )
merged_df.to_csv('../data/merged_df.csv', index=False)
print("Merge counts:")
print(merged_df['_merge'].value_counts())

tp = (merged_df['_merge'] == 'both').sum()
fp = (merged_df['_merge'] == 'left_only').sum()
fn = (merged_df['_merge'] == 'right_only').sum()

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

fp_df = merged_df[merged_df['_merge'] == 'left_only']
print("\nFalse positives (detected sentences without ground truth):")
print(fp_df[['sentence', 'dataset_id']].head())


fn_df = merged_df[merged_df['_merge'] == 'right_only']
print("\nFalse negatives (ground truth missed by detection):")
print(fn_df[['article_id', 'dataset_id']].head())
