import pandas as pd

df = pd.read_csv('Medicine_Details.csv')
json_data = df.to_json(orient='records')
with open('Medicine_Details.json', 'w') as f:
    f.write(json_data)
