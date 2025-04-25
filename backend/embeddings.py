from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import numpy as np


class Embeddings(TfidfVectorizer):
    def __init__(self, documents):
        super().__init__(max_features=5000, stop_words='english', max_df=0.9, min_df=5)
        self.svd_module = TruncatedSVD(n_components=11, random_state=42)
        self.documents = documents

        self.term_document_matrix = self.generate_term_doc_matrix(documents)
        self.svd_matrix = self.singular_value_decomposition(documents)
        self.document_embeddings = self.generate_document_embeddings()

    def generate_term_doc_matrix(self, documents):
        return self.fit_transform(documents['uses'].fillna(""))

    def aggregate_reviews(self, reviews):
        if isinstance(reviews, list):
            return " ".join(review["comment"] for review in reviews if "comment" in review)
        else:
            return ""

    def singular_value_decomposition(self, documents, n_components=1, use_ratings=False):
        review_tfidf = self.transform(documents['user_reviews'].apply(self.aggregate_reviews))
        if use_ratings:
            ratings = np.array(documents['user_ratings'].tolist()).reshape(-1, 1)
            review_tfidf = review_tfidf.multiply(ratings)
        return self.svd_module.fit_transform(review_tfidf)

    def get_term_document_matrix(self):
        return self.term_document_matrix

    def get_SVD_matrix(self):
        return self.svd_matrix

    def get_document_embeddings(self):
        return self.document_embeddings

    def generate_document_embeddings(self):
        return hstack([self.term_document_matrix, csr_matrix(self.svd_matrix)])

    def forward(self, x):
        x_vec = self.transform(x)
        x_decomposed = self.svd_module.transform(x_vec).reshape(1, -1)
        return hstack([x_vec, csr_matrix(x_decomposed)])
    
    def top_svd_terms(self, latent_row, n=3):
        w = self.svd_module.components_.T @ latent_row
        idx = w.argsort()[-n:][::-1]
        return [self.get_feature_names_out()[i] for i in idx]