from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, embeddings, data_loader, clusterer):
        self.embeddings = embeddings
        self.data_loader = data_loader
        self.clusterer = clusterer

    def generate_recommendations(self, query, k, price_filter=None, rating_filter=None, allergy_filter=None):
        query_vector = self.embeddings.forward([query])
        query_vector = self.clusterer.rocchio(q0=query_vector, alpha=98/100, beta=1/100, gamma=1/100)
        A = self.embeddings.document_embeddings
        sims = cosine_similarity(query_vector, A).flatten()
        valid_indices = [i for i, sim in enumerate(sims) if sim > 0]
        valid_indices_sorted = sorted(valid_indices, key=lambda i: sims[i], reverse=True)
        top_indices = valid_indices_sorted[:min(len(valid_indices_sorted), k)]
        recommendations = self.data_loader.retrieve_documents(top_indices)
        recommendations['similarity'] = [sims[i] for i in top_indices]
        
        if allergy_filter:
            recommendations = recommendations[
                ~recommendations['warnings'].str.contains(allergy_filter, case=False, na=False)
            ]

        if price_filter:
            def extract_price(price_info):
                if not price_info or not price_info.get("price"):
                    return None
                price_str = price_info.get("price", "")
                price_str = price_str.replace("$", "").replace(",", "").strip()
                try:
                    return float(price_str)
                except Exception:
                    return None
            recommendations = recommendations[recommendations['price_info'].apply(lambda x: bool(x and x.get("price")))]

            recommendations['price_val'] = recommendations['price_info'].apply(extract_price)
            recommendations['price_val'] = recommendations['price_val'].fillna(float('inf'))
            
            ascending_order = True if price_filter.lower() == "low" else False
            recommendations = recommendations.sort_values(by='price_val', ascending=ascending_order).reset_index(drop=True)

        if rating_filter:
            def extract_rating(user_rating):
                try:
                    return float(str(user_rating).split('/')[0])
                except Exception as e:
                    return 0.0
            recommendations['rating_val'] = recommendations['user_rating'].apply(extract_rating)
            ascending_rating = True if rating_filter.lower() == "low" else False
            recommendations = recommendations.sort_values(by='rating_val', ascending=ascending_rating)

        return recommendations
