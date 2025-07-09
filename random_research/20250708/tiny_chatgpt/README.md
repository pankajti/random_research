To implement a small transformer-based language model from scratch using PyTorch, trained on a synthetic dataset of prompt–response conversations composed exclusively
from a restricted vocabulary of five core English words (I, home, sea, go, eat) and a constrained set of auxiliary verbs and pronouns, while maintaining standard English grammar rules. The goal is to explore how a language model learns structure, compositionality, and conversational ability within a tightly limited linguistic space.

🔍 Key Features
## Restricted Lexicon: 
Only five main words, with grammatical
support from auxiliaries (e.g., "am", "will", "to") 
and pronouns (e.g., "it", "my").

## Standard English Grammar:
All generated sentences conform to correct English syntax, tense, and structure.

## Synthetic Chat Dataset:
Automatically generated English sentence pairs (prompt/response) for supervised training.

## Transformer from Scratch: 
Custom-built GPT-style model using raw PyTorch (no external LM libraries).

## Minimal Language Research Tool:
Designed to investigate how neural models internalize grammar and meaning with minimal input diversity.