from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, embeddings, data_loader):
        self.embeddings = embeddings
        self.data_loader = data_loader

    def generate_recommendations(self, query, k):
        query_vector = self.embeddings.transform([query])
        sims = cosine_similarity(query_vector, self.embeddings.get_term_document_matrix()).flatten()
        top_indices = sims.argsort()[::-1][0:k]
        recommendations = self.data_loader.retrieve_documents(top_indices)
        recommendations['similarity'] = sims[top_indices]
        return recommendations

