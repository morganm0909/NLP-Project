import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def clean_text(text):
    """Remove special characters, extra spaces, and normalize text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters (keep periods)
    return text.strip()

def tokenize_sentences(text):
    """Tokenize text into sentences."""
    return sent_tokenize(text)

def tokenize_words(text):
    """Tokenize text into words."""
    return word_tokenize(text)
