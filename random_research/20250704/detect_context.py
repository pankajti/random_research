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


def main():
    example_text = (
        "In this work, we used the CIFAR-10 dataset for training. "
        "Our experiments were also compared with results from ImageNet. "
        "We implemented the model in TensorFlow. "
        "Future work may include using new data sources. "
        "This is an example sentence without any dataset. "
        "Further details can be found in Smith et al. (2023)."
    )
    detect_candidate_sentences(example_text)

    complex_text = (
        "The COCO dataset (Common Objects in Context), as described by Lin et al. (2014), "
        "is frequently employed. Is MNIST a good choice? We believe so! Results based on "
        "the PASCAL VOC dataset were also considered."
    )
    print("\n--- Running with complex text ---")
    detect_candidate_sentences(complex_text)

if __name__ == "__main__":
    main()