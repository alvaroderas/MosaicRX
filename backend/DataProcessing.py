import pandas as pd
df = pd.read_csv("Medicine_Details.csv")
open('init.json', 'w').write(df.to_json(orient='records'))
DATASET_HEADERS = df.columns.tolist()

