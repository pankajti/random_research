import os
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tokenizer for your NER model
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --------------- Helpers ----------------

def extract_text_from_xml(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    text_parts = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
    return " ".join(text_parts)

def split_into_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text.strip())

def detect_candidate_sentences(text: str, llm) -> List[str]:
    prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=("You are an expert at analyzing scientific papers. "
                  "Does the following sentence contain or refer to a dataset mention or citation? "
                  "Answer with 'Yes' or 'No'.\n\n"
                  "Sentence: \"{sentence}\"")
    )
    chain = prompt_template | llm.bind(stop=["\n", " "])

    sentences = split_into_sentences(text)
    candidate_sentences = []
    for i, sentence in enumerate(sentences):
        response = chain.invoke({"sentence": sentence}).content.strip().lower()
        print(f"Sentence {i+1}: {sentence}\nResponse: {response}\n")
        if response.startswith("yes"):
            candidate_sentences.append(sentence)
    return candidate_sentences

def create_bio_labels(tokens: List[str], mention: str) -> List[int]:
    mention_tokens = tokenizer.tokenize(mention)
    bio_labels = [0] * len(tokens)
    joined_tokens = " ".join(tokens).lower()
    joined_mention = " ".join(mention_tokens).lower()
    start_idx = joined_tokens.find(joined_mention)
    if start_idx == -1:
        return bio_labels  # mention not found
    mention_start = len(joined_tokens[:start_idx].split())
    for i in range(len(mention_tokens)):
        bio_labels[mention_start + i] = 1 if i == 0 else 2  # 1=B, 2=I
    return bio_labels

# --------------- Main Pipeline ----------------

def main():
    data_path = r'/Users/pankajti/dev/data/kaggle/make-data-count-finding-data-references'
    train_labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    ollama_model = "llama3.1"
    llm = ChatOllama(model=ollama_model)

    all_inputs, all_labels = [], []

    for idx, rec in train_labels[:1].iterrows():
        xml_path = os.path.join(data_path, 'train', 'XML', f"{rec['article_id']}.xml")
        if not os.path.exists(xml_path):
            continue

        full_text = extract_text_from_xml(xml_path)
        candidate_sentences = detect_candidate_sentences(full_text, llm)

        for sent in candidate_sentences:
            mention = rec['dataset_id'].split('/')[-1].lower()
            if mention in sent.lower():
                tokens = tokenizer.tokenize(sent)
                bio_labels = create_bio_labels(tokens, mention)
                tokenized = tokenizer(sent, truncation=True, padding='max_length', max_length=256)
                tokenized['labels'] = bio_labels + [-100] * (len(tokenized['input_ids']) - len(bio_labels))
                all_inputs.append(tokenized)

    dataset = Dataset.from_list(all_inputs)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)  # 0=O, 1=B, 2=I
    args = TrainingArguments(
        output_dir="./scibert_ner_output",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

if __name__ == "__main__":
    nltk.download('punkt')
    main()
