import pandas as pd

df = pd.read_csv("Medicine_Details.csv")
DATASET_HEADERS = df.columns.tolist()
open('init.json', 'w').write(df.to_json(orient='records'))


