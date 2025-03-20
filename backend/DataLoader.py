import json
from DataProcessing import DATASET_HEADERS

class DataLoader:
    def __init__(self, json_file: str, batch_size: int = 32):
        """
        Initialize the DataLoader with a JSON file and batch size.
        """
        self.json_file = json_file
        self.batch_size = batch_size
        self.dataset = self._load_data()
        self.dataset_headers = DATASET_HEADERS

    def _load_data(self):
        """
        Loads the JSON file into a list of dictionaries and extracts headers.
        """
        return json.load(open(self.json_file, 'r'))
    


