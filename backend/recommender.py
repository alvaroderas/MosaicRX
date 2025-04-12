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
        valid_indices = [i for i, sim in enumerate(sims) if sim > 0]
        if not valid_indices:
            return self.data_loader.retrieve_documents([])
        valid_indices_sorted = sorted(valid_indices, key=lambda i: sims[i], reverse=True)
        top_indices = valid_indices_sorted[:min(len(valid_indices_sorted), k)]
        recommendations = self.data_loader.retrieve_documents(top_indices)
        recommendations['similarity'] = sims[top_indices]
        return recommendations

