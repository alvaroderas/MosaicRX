from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, dataloader, embeddings, k=10):
        self.dataloader = dataloader
        self.embeddings = embeddings
        self.term_document_matrix = self.embeddings.__generate_term_doc_matrix__()
        self.top_k = k

    def __optimize_query__(self, query):
        return query

    def __recommendations__(self, query_vector):
        sims = cosine_similarity(query_vector, self.term_document_matrix).flatten()
        top_indices = sims.argsort()[::-1][:self.top_k]
    
        recommendations = {}
        for idx in top_indices:
            recommendations[idx] = {
                "data": self.dataloader.retrieve_document(doc_id=idx),
                "similarity": sims[idx]
            }

        return recommendations

    def search(self, query):
        q_vector = self.embeddings.transform(query)
        q_vector = self.__optimize_query__(q_vector)
        return self.__recommendations__(q_vector)
