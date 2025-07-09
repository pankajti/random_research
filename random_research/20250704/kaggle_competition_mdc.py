
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

import pandas as pd
import os

import xml.etree.ElementTree as ET


import re
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import List
import nltk

try:
    # Ensure NLTK 'punkt' tokenizer data is downloaded
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')

# Use NLTK for robust sentence tokenization
nltk_available = True # Assume NLTK is installed after the check/download


def split_into_sentences(text: str) -> List[str]:
    """
    Splits a given text into individual sentences using NLTK's sent_tokenize.
    """
    return nltk.tokenize.sent_tokenize(text.strip())


def detect_candidate_sentences(
    article_text: str,
    ollama_model: str = "llama3.1",
    prompt_template_str: str = (
        "You are an expert at analyzing scientific papers. "
        "Does the following sentence contain or refer to a dataset mention or citation? "
        "Answer with 'Yes' or 'No'.\n\n"
        "Sentence: \"{sentence}\""
    )
) -> List[str]:
    """
    Detects sentences that likely contain or refer to a dataset mention or citation
    using a local Ollama LLM.

    Args:
        article_text: The full text of the article to analyze.
        ollama_model: The name of the Ollama model to use (e.g., "llama3.1").
        prompt_template_str: The string template for the LLM prompt.

    Returns:
        A list of sentences identified as containing dataset mentions.
    """
    print(f"Attempting to load Ollama model: {ollama_model}")
    try:
        llm = ChatOllama(model=ollama_model)
    except Exception as e:
        print(f"Error loading Ollama model '{ollama_model}'. Please ensure it's running and the model is pulled.")
        print(f"Error details: {e}")
        return []

    prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=prompt_template_str
    )

    # FIX: Explicitly convert the PromptValue to a string before passing to LLM
    # The .str() method on a PromptTemplate's output ensures it's a plain string.
    chain = prompt_template | llm.bind(stop=["\n", " "]) # Added stop sequence to encourage concise answers
    # Note: For some LLMs, using .str() is implicitly handled, but for Ollama or certain
    # versions of LangChain, making it explicit can resolve this 'input' error.

    sentences = split_into_sentences(article_text)
    candidate_sentences = []

    print("\n--- Analyzing Sentences ---")
    for i, sentence in enumerate(sentences):
        try:
            # When you invoke a chain, you pass the input for the *first* component.
            # Here, 'sentence' is the input for prompt_template.
            response = chain.invoke({"sentence": sentence}).content.strip().lower() # Ensure input is a dictionary
            print(f"Sentence {i+1}: \"{sentence}\"\nLLM Response: \"{response}\"\n")
            if response.startswith("yes"):
                candidate_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence \"{sentence}\" with LLM: {e}")
            print("Skipping this sentence.")

    print("\n=== Candidate Sentences Detected ===")
    if not candidate_sentences:
        print("No candidate sentences found.")
    else:
        for s in candidate_sentences:
            print(f"- {s}")

    return candidate_sentences

def extract_text_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    text_parts = []

    # Iterate through sections likely containing text
    for elem in root.iter():
        if elem.text:
            clean_text = elem.text.strip()
            if clean_text:
                text_parts.append(clean_text)

    # Recombine into a single string for offset-based analysis
    full_text = " ".join(text_parts)
    return full_text



def main():
    data_path = r'/Users/pankajti/dev/data/kaggle/make-data-count-finding-data-references'
    files = os.listdir(data_path)
    train_labels_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    for idx , rec in train_labels_df[:1].iterrows():
        xml_path = os.path.join(data_path,'train','XML', rec['article_id']+".xml")
        if os.path.exists(xml_path):
            print(xml_path)
            full_text = extract_text_from_xml(xml_path)
            print(full_text)
            mention_text= rec['dataset_id']

            detect_candidate_sentences(full_text)

            found_idx = full_text.find(mention_text[-10:])
            if found_idx != -1:
                print(f"\nFound mention at char index: {found_idx}")
                print(f"Context: ...{full_text[max(0, found_idx - 30):found_idx + len(mention_text) + 30]}...")
            else:
                print("\nMention NOT found in extracted text.")

        print(idx, rec)
        # output of above 0 article_id              10.1002_2017jc013030
        # dataset_id    https://doi.org/10.17882/49388
        # type                                 Primary
        # Name: 0, dtype: object


    #load_model()


if __name__ == "__main__":
    main()

def load_model():
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    # TODO: your data prep, Trainer setup, etc.
    print(model)
