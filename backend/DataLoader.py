import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Generator, Tuple
import pandas as pd
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

    def _load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Loads the JSON file into a list of dictionaries and extracts headers.
        """
        json_data = json.load(open(self.json_file, 'r'))
        return json_data


# Example usage
if __name__ == "__main__":
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init.json')
    data_loader = DataLoader(json_file=json_file_path, batch_size=16)
