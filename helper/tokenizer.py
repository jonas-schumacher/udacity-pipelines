import re
from typing import List

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize(text: str) -> List[str]:
    """
    Tokenize a given sentence by normalizing, removing stopwords and lemmatizing.

    Parameters
    ----------
    text: str
        given sentence

    Returns
    -------
    List[str]
        tokenized sentence
    """
    # Normalize text
    normalized_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(normalized_text)

    # Remove stop words
    words_cleaned = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(w) for w in words_cleaned]
    words_lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in words_lemmatized]

    return words_lemmatized
