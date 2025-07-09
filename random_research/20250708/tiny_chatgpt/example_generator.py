import random
import json
from pathlib import Path

# Define the fixed vocabulary
core_words = ["I", "home", "sea", "go", "eat"]
auxiliary_words = [
    "am", "is", "was", "are", "will", "have", "has", "to", "at", "the", "it", "me", "my", "you", "where", "do", "did"
]

# Create the complete vocabulary
vocab = sorted(set(core_words + auxiliary_words + [".", "?"]))
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

# Generate synthetic sentence pairs
def generate_sentence_pairs(n=1000):
    sentence_pairs = []
    for _ in range(n):
        subject = random.choice(["I", "you"])
        verb = random.choice(["go", "eat"])
        aux = random.choice(["will", "have", "am", "did", ""])
        loc = random.choice(["home", "sea"])
        preposition = random.choice(["to", "at"])
        pronoun = random.choice(["my", "it", "me"])
        question = random.choice(["Where do I go?", "What do I eat?", "Do I go home?", "Did I eat it?"])

        # Form statement-response pairs
        if random.random() < 0.5:
            sentence = f"{subject} {aux} {verb} {preposition} the {loc}.".strip()
            response = f"{subject} {aux} {verb} at {loc}.".strip()
        else:
            sentence = question
            response = f"{subject} {aux} {verb} {preposition} {loc}.".strip()

        sentence_pairs.append({"prompt": sentence, "response": response})
    return sentence_pairs

# Generate and save the dataset
dataset = generate_sentence_pairs(1000)
dataset_path = Path("data/mini_language_chat_dataset.json")
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=2)
