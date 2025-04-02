import pandas as pd
import json

class DataLoader:
    def __init__(self, json_file):
        self._json_file = json_file
        self._dataset = self._load_data()
        self._dataset_headers = self._dataset.columns.tolist()
        self._dataset_size = len(self._dataset)

    def _load_data(self):
        return pd.DataFrame(json.load(open(self._json_file, 'r')))

    def json_file(self):
        return self._json_file

    def dataset(self):
        return self._dataset

    def headers(self):
        return self._dataset_headers

    def retrieve_documents(self, doc_ids):
        return self.dataset().iloc[doc_ids].copy()
