import spacy
from spacy.tokens import Doc


class NaturalLanguageProcessor:
    """
    A class for natural language processing tasks.
    """
    instance = None

    def __init__(self):
        self._spacy_nlp = spacy.load("en_core_web_trf")

    @classmethod
    def get(cls):
        """
        Returns the loaded spaCy NLP model.
        """
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def process(self, text: str) -> Doc:
        """
        Processes the input text and returns a spaCy Doc object.
        """
        return self._spacy_nlp(text)
