
import pandas as pd
import json

class DataLoader:
    def __init__(self, json_file: str, batch_size: int = 32):
        """
        Initialize the DataLoader with a JSON file and batch size.
        """
        self._json_file = json_file
        self._batch_size = batch_size
        self._dataset = self._load_data()
        self._dataset_headers = self._dataset.columns.tolist()
        self._dataset_size = len(self._dataset)

    def _load_data(self):
        """
        Loads the JSON file into a pandas DataFrame.
        """
        return pd.DataFrame(json.load(open(self._json_file, 'r')))

    def json_file(self):
        return self._json_file

    def batch_size(self):
        return self._batch_size

    def dataset(self):
        return self._dataset

    def headers(self):
        return self._dataset_headers

    def retrieve_documents(self, doc_ids):
        """
        Retrieve multiple documents by their indices.
        """
        return self.dataset().iloc[doc_ids].copy()
