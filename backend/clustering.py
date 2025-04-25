from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

votes = defaultdict(int)

def register_vote(name, delta):
    votes[name] += delta

class Clustering(KMeans):
    def __init__(self, doc_emb):
        super().__init__(n_clusters=15, init="random", n_init=10, max_iter=500, random_state=43)
        self.doc_emb = doc_emb
        self.fit(doc_emb)

    def rocchio(self, q0, alpha=.9, beta=.05, gamma=.05):
        q0_vec = q0.toarray().ravel()
        pos = [self.doc_emb[i].toarray().ravel() for i, v in votes.items() if v > 0]
        neg = [self.doc_emb[i].toarray().ravel() for i, v in votes.items() if v < 0]
        pos_vec = np.mean(pos, axis=0) if pos else 0
        neg_vec = np.mean(neg, axis=0) if neg else 0
        return (alpha*q0_vec + beta*pos_vec - gamma*neg_vec).reshape(1, -1)
