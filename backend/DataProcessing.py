import pandas as pd
import json


with open('init.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
DATASET_HEADERS = df.columns.tolist()

print(DATASET_HEADERS)
