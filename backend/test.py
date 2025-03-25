from Embeddings import *
from DataLoader import *
from Recommender import *
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

data_loader = DataLoader(json_file=json_file_path)
embeddings = Embeddings(documents=data_loader.dataset())
irsys = Recommender(embeddings=embeddings, data_loader=data_loader)

queries = ["headache", "sickness","fever", " i am having a cold and fever and need medicine"]
for query in queries:
    result = irsys.generate_recommendations(query, 3)
    print(result)





