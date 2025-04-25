from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender:
    def __init__(self, embeddings, data_loader, clusterer=None):
        self.emb  = embeddings
        self.data = data_loader
        self.A    = embeddings.get_document_embeddings()

    def generate_recommendations(self, query, k, price_filter=None, rating_filter=None, allergy_filter=None):
        q_full = self.emb.forward([query])
        tfidf_dim = self.emb.term_document_matrix.shape[1]
        sims_tf = cosine_similarity(q_full[:, :tfidf_dim], self.A[:,  :tfidf_dim]).flatten()
        pos_mask = sims_tf > 0
        if not np.any(pos_mask):
            return self.data.retrieve_documents([])

        seed_idx = np.where(pos_mask)[0]
        sims_svd = cosine_similarity(q_full[:, tfidf_dim:], self.A[seed_idx, tfidf_dim:]).flatten()
        final = 0.7 * sims_tf[seed_idx] + 0.3 * sims_svd
        order = final.argsort()[::-1]
        top = seed_idx[order[:k]]

        recs = self.data.retrieve_documents(top).reset_index(drop=True)
        recs["similarity"] = final[order[:k]]
        recs["tags"] = [self.emb.top_svd_terms(self.emb.svd_matrix[i], 5) for i in top]
        recs["snippet"] = ""

        for pos, idx in enumerate(top):
            reviews = recs.at[pos, "user_reviews"]
            if isinstance(reviews, list):
                for r in reviews:
                    if r.get("rating", "").startswith(tuple("78910")):
                        recs.at[pos, "snippet"] = r["comment"][:180] + "…"
                        break

        if allergy_filter:
            recs = recs[~recs["warnings"].str.contains(allergy_filter, na=False, case=False)]

        if price_filter:
            def get_price(p):
                if isinstance(p, dict):
                    price = str(p.get("price", "")).strip()
                    if price.startswith("$"):
                        try:
                            return float(price[1:].replace(",", ""))
                        except ValueError:
                            pass
                return np.nan

            recs["price_val"] = recs["price_info"].apply(get_price)
            recs = recs.dropna(subset=["price_val"])

            ascending = price_filter.lower() == "low"
            recs = recs.sort_values("price_val", ascending=ascending, na_position="last").reset_index(drop=True)

        if rating_filter:
            recs["rating_val"] = recs["user_rating"].str.split("/").str[0].astype(float)
            recs = recs.sort_values("rating_val", ascending=rating_filter == "low")

        return recs
