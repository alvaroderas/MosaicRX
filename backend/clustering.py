from sklearn.cluster import KMeans
import numpy as np

class Clustering(KMeans):
    def __init__(self,document_embeddings, n_clusters=15, init='random', n_init=10, max_iter=500, tol=1e-4, random_state=0):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        self.document_embeddings = document_embeddings

        self.fit(document_embeddings)
        self.clusters = self.cluster_centers_
        self.centroids = {idx: centroid for idx, centroid in enumerate(self.cluster_centers_)}

    def forward(self, x):
        return self.predict(x.reshape(1, -1))[0]


    def rocchio(self, q0, alpha=0.9, beta=0.05, gamma=0.05):
        q0 = q0.toarray().flatten()
        relevant_cluster = self.forward(q0)
        relevant_centroid = self.centroids[relevant_cluster]
        non_relevant_centroids = np.sum(
            [self.centroids[idx] for idx in self.centroids if idx != relevant_cluster],
            axis=0
        )
        q_opt = alpha*q0 + beta*relevant_centroid - gamma*non_relevant_centroids
        return q_opt.reshape(1, -1)

