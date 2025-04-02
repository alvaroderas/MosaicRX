import json
import pandas as pd

predicate = lambda n: n.get("user_rating") != "" and n.get("user_reviews") != []


with open('init.json', 'r') as f:
    dataset = json.load(f)
    filtered_dataset = list(filter(predicate, dataset))

with open('inittest.json', 'w') as f:
    json.dump(filtered_dataset, f, indent=4)


df = pd.DataFrame(dataset)
DATASET_HEADERS = df.columns.tolist()


