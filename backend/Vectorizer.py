from collections.abc import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from DataLoader import DataLoader
from typing import List, Tuple

class Vectorizer:
    def __init__(self,
                 dataloader: DataLoader,
                 tokenizer: Callable,
                 text_field: str = None,
                 max_features: int = 5000,
                 stop_words: str = 'english',
                 max_df: float = 0.8,
                 min_df: int = 10,
                 norm: str = 'l2'):
        """
        Initialize the Vectorizer.
        
        Parameters
        ----------
        dataloader : DataLoader
            An instance of DataLoader with attributes `dataset` and `dataset_headers`.
        tokenizer : Callable
            A tokenizer function (or class) that converts text to tokens.
        text_field : str, optional
            The field name (from dataloader.dataset_headers) containing the text.
            If not provided, the class will try to infer it from the headers.
        max_features : int, optional
            Maximum number of features for TF-IDF.
        stop_words : str, optional
            Stop words to be used by the TF-IDF vectorizer.
        max_df : float, optional
            Max document frequency for the TF-IDF vectorizer.
        min_df : int, optional
            Min document frequency for the TF-IDF vectorizer.
        norm : str, optional
            Norm used in the TF-IDF vectorizer.
        """
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        # Infer text field from dataloader.dataset_headers if not provided.
        self.text_field = text_field if text_field is not None else self._infer_text_field(dataloader.dataset_headers)
        
        # Build the TF-IDF vectorizer using the provided tokenizer and parameters.
        self.tfidf_vectorizer = self.build_vectorizer(max_features, stop_words, max_df, min_df, norm)
        
        # Build a corpus from the dataset based on the chosen text field.
        self.corpus = self._build_corpus()
        # Fit and transform to get the document-term matrix.
        self.doc_term_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        # Store the vocabulary (i.e., feature names).
        self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
    
    def build_vectorizer(self, max_features: int, stop_words: str, max_df: float, min_df: int, norm: str) -> TfidfVectorizer:
        """
        Build and return a TfidfVectorizer with the given parameters, using the provided tokenizer.
        """
        return TfidfVectorizer(tokenizer=self.tokenizer,
                               max_features=max_features,
                               stop_words=stop_words,
                               max_df=max_df,
                               min_df=min_df,
                               norm=norm)
    
    def _infer_text_field(self, headers: List[str]) -> str:
        """
        Infer which header to use for text by looking for common keywords.
        If none match, the first header is returned.
        """
        candidates = [h for h in headers if any(kw in h.lower() for kw in ['text', 'description', 'content'])]
        if candidates:
            return candidates[0]
        elif headers:
            return headers[0]
        else:
            raise ValueError("No headers available to infer text field.")
    
    def _build_corpus(self) -> List[str]:
        """
        Build a corpus (list of strings) from the dataset using the selected text field.
        """
        corpus = []
        for record in self.dataloader.dataset:
            text = record.get(self.text_field, '')
            corpus.append(str(text))
        return corpus
    
    def transform(self, new_texts: List[str]):
        """
        Transform new texts using the fitted TF-IDF vectorizer.
        """
        return self.tfidf_vectorizer.transform(new_texts)
    
    def get_doc_term_matrix(self):
        """
        Return the document-term matrix computed during initialization.
        """
        return self.doc_term_matrix

# Example usage:
if __name__ == "__main__":
    import os
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init.json')
    data_loader = DataLoader(json_file=json_file_path, batch_size=16)
    
    # Use TfidfVectorizer's tokenizer by providing a callable. For example, a simple whitespace tokenizer:
    tokenizer = lambda text: text.split()
    
    vectorizer = Vectorizer(dataloader=data_loader, tokenizer=tokenizer)
    
    # Print the shape of the document-term matrix.
    print(vectorizer.get_doc_term_matrix().shape)

    # Print the inferred vocabulary.
    print(vectorizer.vocabulary)
