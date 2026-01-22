import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


_STOP_WORDS = None
_LEMMATIZER = None


def ensure_nltk_data():
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_total_text(df):
    df = df.copy()
    df["description"] = df["description"].fillna("no description")
    df["skills_desc"] = df["skills_desc"].fillna("no skills desc")
    df["title"] = df["title"].fillna("no title")
    df["total_text"] = (
        df["description"] + " " + df["skills_desc"] + " " + df["title"]
    )
    return df


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    return [word for word in tokens if word not in _STOP_WORDS]


def lemmatize_tokens(tokens):
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return [_LEMMATIZER.lemmatize(word) for word in tokens]


def tokens_to_text(tokens):
    return " ".join(tokens)


def preprocess_series(series):
    cleaned = series.apply(clean_text)
    tokenized = cleaned.apply(tokenize)
    no_stopwords = tokenized.apply(remove_stopwords)
    lemmatized = no_stopwords.apply(lemmatize_tokens)
    return lemmatized.apply(tokens_to_text)
