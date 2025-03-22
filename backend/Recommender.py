from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self,  embeddings):
        self.embeddings = embeddings
        self.data_loader = embeddings.data_loader
        self.term_document_matrix = self.embeddings.__generate_term_doc_matrix__()
    

    def __optimize_query__(self, query):
        return query

    def __recommendations__(self, query_vector, top_k):
        sims = cosine_similarity(query_vector, self.term_document_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
    
        recommendations = {}
        for idx in top_indices:
            recommendations[idx] = {
                "data": self.data_loader.retrieve_document(doc_id=idx),
                "similarity": sims[idx]
            }

        return recommendations

    def search(self, query, k=10):
        q_vector = self.embeddings.transform(query)
        q_vector = self.__optimize_query__(q_vector)
        return self.__recommendations__(q_vector, k)
