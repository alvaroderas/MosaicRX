from embeddings import Embeddings
from data_loader import DataLoader
from recommender import Recommender
from clustering import Clustering
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'inittest.json')
data_loader = DataLoader(json_file=json_file_path)
documents = data_loader.dataset()
embeddings = Embeddings(documents=documents)
clusterer = Clustering(document_embeddings=embeddings.get_document_embeddings())
irsys = Recommender(embeddings=embeddings, data_loader=data_loader, clusterer=clusterer)


# Example queries
queries = ["headache", "sickness", "fever", "I am having a cold and fever and need medicine"]

for query in queries:
    t0 = time.time()
    recommendations = irsys.generate_recommendations(query,k=5)
    tf = time.time()
    print(f"Query: {query}")
    print(recommendations[['uses', 'similarity']])
    print(f"Execution time: {tf - t0} seconds")
    print("=" * 50)
