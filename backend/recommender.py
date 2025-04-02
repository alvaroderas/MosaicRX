from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, embeddings, data_loader, clusterer):
        self.embeddings = embeddings
        self.data_loader = data_loader
        self.clusterer = clusterer

    def generate_recommendations(self, query, k):
        query_vector = self.embeddings.forward([query])
        query_vector = self.clusterer.rocchio(q0=query_vector, alpha=98/100, beta=1/100, gamma=1/100)
        A = self.embeddings.document_embeddings
        sims = cosine_similarity(query_vector, A).flatten()
        top_indices = sims.argsort()[::-1][0:k]
        recommendations = self.data_loader.retrieve_documents(top_indices)
        recommendations['similarity'] = sims[top_indices]
        return recommendations

