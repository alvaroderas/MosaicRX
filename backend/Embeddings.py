from collections.abc import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from DataLoader import DataLoader
from typing import List, Tuple

class Embeddings:
    def __init__(self,
                 dataloader: DataLoader,
                 tokenizer: Callable,
                 max_features: int = 5000,
                 stop_words: str = 'english',
                 max_df: float = 0.8,
                 min_df: int = 10,
                 norm: str = 'l2'):
        """
        Computes an embedding for each dict in dataloader
        
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
   
    
    def build_embeddings(self):
        U = []
        for idx, data_instance in enumerate(self.dataloader.dataset):
            print(idx, data_instance)
        return U 

    