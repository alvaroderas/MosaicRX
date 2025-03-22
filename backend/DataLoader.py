import json
from DataProcessing import DATASET_HEADERS

class DataLoader:
    def __init__(self, json_file: str, batch_size: int = 32):
        """
        Initialize the DataLoader with a JSON file and batch size.
        """
        self._json_file = json_file
        self._batch_size = batch_size
        self._dataset = self._load_data()
        self._dataset_headers = DATASET_HEADERS

    def _load_data(self):
        """
        Loads the JSON file into a list of dictionaries and extracts headers.
        """
        return json.load(open(self._json_file, 'r'))

  
    def json_file(self):
        return self._json_file

    def batch_size(self):
        return self._batch_size

    def dataset(self):
        return self._dataset

    def dataset_headers(self):
        return self._dataset_headers

    def retrieve_document(self, doc_id):
        return self._dataset[doc_id]
    
    def retrieve_documents(self, doc_ids):
        documents = []
        for doc_id in doc_ids:
            documents.append(self.retrieve_document(doc_id=doc_id))
        return documents

    


