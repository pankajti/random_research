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


import torch

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from seqeval.metrics import classification_report as seqeval_classification_report # For NER metrics


# --- NER Model Training ---
model_ner = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(ner_label_map),
    id2label=id_to_label_ner,
    label2id=ner_label_map
)

data_collator_ner = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define NER metrics
def compute_metrics_ner(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (where label is -100)
    true_labels = [[id_to_label_ner[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label_ner[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args_ner = TrainingArguments(
    output_dir="./results_ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_ner",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none" # Disable reporting to Hugging Face Hub if not needed
)

trainer_ner = Trainer(
    model=model_ner,
    args=training_args_ner,
    train_dataset=ner_train_dataset,
    eval_dataset=ner_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator_ner,
    compute_metrics=compute_metrics_ner,
)

print("\n--- Starting NER Training ---")
# trainer_ner.train()
# print(trainer_ner.evaluate())


# --- Classification Model Training ---
model_classification = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(classification_label_map),
    id2label=id_to_label_classification,
    label2id=classification_label_map
)

# Define Classification metrics
def compute_metrics_clf(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args_classification = TrainingArguments(
    output_dir="./results_classification",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_classification",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none" # Disable reporting to Hugging Face Hub if not needed
)

trainer_classification = Trainer(
    model=model_classification,
    args=training_args_classification,
    train_dataset=clf_train_dataset,
    eval_dataset=clf_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_clf, # Data collator is not strictly needed if tokenizer handles padding
)

print("\n--- Starting Classification Training ---")
# trainer_classification.train()
# print(trainer_classification.evaluate())