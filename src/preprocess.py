import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer


class TextPreprocessor:
    def __init__(self):
        # Use sklearn's built-in stopwords (NO NLTK DOWNLOAD REQUIRED)
        self.stop_words = set(ENGLISH_STOP_WORDS)

        # Stemming still works without NLTK downloads
        self.stemmer = PorterStemmer()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text for ML model.
        """

        if not isinstance(text, str):
            return ""

        # lowercase
        text = text.lower()

        # remove everything except letters
        text = re.sub(r'[^a-z\s]', '', text)

        words = text.split()

        cleaned_words = [
            self.stemmer.stem(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(cleaned_words)