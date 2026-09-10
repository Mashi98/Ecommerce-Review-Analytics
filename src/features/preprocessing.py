"""Text preprocessing: tokenization, POS-aware lemmatization, stopword removal.

Uses NLTK (punkt tokenizer, averaged_perceptron_tagger for POS tags,
WordNet for lemmatization). Run once per environment:

    import nltk
    for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg)
"""

import re
import string

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))
_punct_table = str.maketrans("", "", string.punctuation)

_POS_MAP = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def _wordnet_pos(tag: str) -> str:
    return _POS_MAP.get(tag[0], wordnet.NOUN)


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, HTML tags, digits, and punctuation."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(_punct_table)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Clean, tokenize, drop stopwords/single characters, and lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tagged = pos_tag(tokens)
    return [
        _lemmatizer.lemmatize(tok, _wordnet_pos(tag))
        for tok, tag in tagged
        if tok not in _stopwords and len(tok) > 1
    ]


def preprocess(text: str) -> str:
    """Full pipeline: returns the cleaned, lemmatized token stream as a string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return " ".join(tokenize(text))
