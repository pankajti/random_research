import os
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tokenizer for your NER model
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

import re

def is_dataset_id_in_sentence(sentence: str, dataset_id: str) -> bool:
    norm_sentence = re.sub(r'\s+', ' ', sentence.lower())
    parts = dataset_id.lower().split('/')

    if len(parts) >= 2:
        norm_id = f"{parts[-2]}/{parts[-1]}"
    else:
        norm_id = dataset_id.lower()
    # get the DOI part
    if norm_id in norm_sentence:
        return True

    # Check for DOI patterns in the sentence directly
    doi_pattern = re.compile(r'10\.\d{4,9}/\S+', re.I)
    if doi_pattern.search(norm_sentence):
        for doi_match in doi_pattern.findall(norm_sentence):
            if norm_id in doi_match.lower():
                return True

    return False

def extract_text_from_xml(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    text_parts = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
    return " ".join(text_parts)

def split_into_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text.strip())




def detect_candidate_sentences(text: str, llm) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detects sentences containing dataset mentions using LLM, and returns:
    - A DataFrame with each sentence + its yes/no flag
    - A list of sentences flagged 'yes' as candidate sentences
    """
    prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=("You are an expert at analyzing scientific papers. "
                  "Does the following sentence contain or refer to a dataset mention or citation? "
                  "Answer with 'Yes' or 'No'.\n\n"
                  "Sentence: \"{sentence}\"")
    )
    chain = prompt_template | llm.bind(stop=["\n", " "])

    sentences = split_into_sentences(text)
    rows = []

    print("\n--- Analyzing Sentences ---")
    for i, sentence in tqdm(enumerate(sentences),"Analysing sentences"):
        try:
            response = chain.invoke({"sentence": sentence}).content.strip().lower()
            #print(f"Sentence {i + 1}: \"{sentence}\"\nLLM Response: \"{response}\"\n")
            flag = "yes" if response.startswith("yes") else "no"
            rows.append({"sentence": sentence, "mention_citation_flag": flag})
        except Exception as e:
            print(f"Error processing sentence \"{sentence}\" with LLM: {e}")
            print("Skipping this sentence.")

    df = pd.DataFrame(rows)
    candidate_sentences = df[df["mention_citation_flag"] == "yes"]["sentence"].tolist()

    print("\n=== Candidate Sentences Detected ===")
    if not candidate_sentences:
        print("No candidate sentences found.")

    return df, candidate_sentences


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

def main():
    data_path = r'/Users/pankajti/dev/data/kaggle/make-data-count-finding-data-references'
    train_labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    ollama_model = "llama3.1"
    llm = ChatOllama(model=ollama_model)


    all_detected_dfs = []
    for idx, rec in train_labels.iterrows():
        article_dataset_ids = train_labels[train_labels["article_id"] == rec["article_id"]]

        xml_path = os.path.join(data_path, 'train', 'XML', f"{rec['article_id']}.xml")
        print(f"running {xml_path}")

        if not os.path.exists(xml_path):
            continue

        full_text = extract_text_from_xml(xml_path)
        sentences_df, candidate_sentences = detect_candidate_sentences(full_text, llm)


        # Collect rows for the dataframe
        rows = []

        for idx, rec in tqdm(sentences_df.iterrows()):
            sentence = rec['sentence']
            llm_flag = rec['mention_citation_flag']
            matching_id = None
            # Iterate over dataset_ids for this article in train_labels
            for dataset_id in article_dataset_ids['dataset_id']:
                match = is_dataset_id_in_sentence(sentence, dataset_id)
                if match:
                    matching_id = dataset_id
                    break

            rows.append({
                "sentence": sentence,
                "dataset_id": matching_id,  # None if no match found
                "llm_flag": llm_flag,
            })

        df = pd.DataFrame(rows)
        detected_df = df[~df.dataset_id.isna()]
        all_detected_dfs.append(detected_df)

    final_df = pd.concat(all_detected_dfs, ignore_index=True)
    final_df.to_csv("all_detected_sentences.csv", index=False)




if __name__ == '__main__':
    main()