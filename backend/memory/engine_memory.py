"""MemoryEngine - Semantic memory with K-Means clustering (numpy-only).

K-Means implemented from scratch with numpy (no sklearn dependency).
Embeddings via Ollama locusai/all-minilm-l6-v2.
"""
import numpy as np
from backend.persistence.database import SessionLocal, Rule
from backend.memory.engine_embeddings import embeddings_engine
from backend.config import KMEANS_N_CLUSTERS, KMEANS_MAX_ITER


class NumpyKMeans:
    """Minimal K-Means clustering using only numpy."""

    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray) -> "NumpyKMeans":
        """Fit K-Means to embedding matrix X (n_samples, n_features)."""
        if len(X) < self.n_clusters:
            # Not enough data for clustering
            self.labels = np.zeros(len(X), dtype=int)
            self.centroids = X if len(X) > 0 else np.array([])
            return self

        # Initialize centroids using k-means++
        self.centroids = self._kmeans_plus_plus_init(X)
        
        for _ in range(self.max_iter):
            # Assign labels
            distances = np.array([
                np.linalg.norm(X - c, axis=1) for c in self.centroids
            ])
            new_labels = np.argmin(distances, axis=0)
            
            # Check convergence
            if self.labels is not None and np.array_equal(self.labels, new_labels):
                break
            self.labels = new_labels
            
            # Update centroids
            for k in range(self.n_clusters):
                members = X[self.labels == k]
                if len(members) > 0:
                    self.centroids[k] = members.mean(axis=0)
        
        return self

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization for better centroid selection."""
        n = len(X)
        centroids = [X[np.random.randint(n)]]
        
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min(np.linalg.norm(x - c) for c in centroids) for x in X
            ])
            probs = distances ** 2 / (distances ** 2).sum()
            next_idx = np.random.choice(n, p=probs)
            centroids.append(X[next_idx])
        
        return np.array(centroids)


class MemoryEngine:
    def __init__(self):
        self.n_clusters = KMEANS_N_CLUSTERS

    def get_all_embeddings(self):
        """Fetch all rules with embeddings from DB."""
        db = SessionLocal()
        rules = db.query(Rule).filter(Rule.embedding != None).all()
        db.close()
        
        if not rules:
            return [], []
        
        embeddings = [embeddings_engine.to_numpy(r.embedding) for r in rules]
        ids = [r.id for r in rules]
        return ids, np.array(embeddings)

    def rebuild_clusters(self):
        """Cluster rules by semantic similarity using numpy K-Means."""
        ids, embeddings = self.get_all_embeddings()
        
        if len(ids) < 2:
            return False, "Not enough rules with embeddings to cluster."
        
        kmeans = NumpyKMeans(n_clusters=min(self.n_clusters, len(ids)), max_iter=KMEANS_MAX_ITER)
        kmeans.fit(embeddings)
        
        # Persist cluster assignments
        db = SessionLocal()
        for i, rule_id in enumerate(ids):
            rule = db.query(Rule).filter(Rule.id == rule_id).first()
            if rule:
                rule.cluster_id = int(kmeans.labels[i])
        db.commit()
        db.close()
        
        return True, f"Clustered {len(ids)} rules into {kmeans.n_clusters} groups."

    def find_similar_rules(self, query_text: str, limit: int = 3):
        """Find rules semantically similar to query via cosine similarity."""
        query_embedding_bytes = embeddings_engine.generate(query_text)
        query_embedding = embeddings_engine.to_numpy(query_embedding_bytes)
        
        ids, embeddings = self.get_all_embeddings()
        
        if not ids or len(embeddings) == 0:
            return []

        # Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(embeddings, query_embedding) / norms
        
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        db = SessionLocal()
        results = []
        for idx in top_indices:
            rule = db.query(Rule).filter(Rule.id == ids[idx]).first()
            if rule:
                results.append({
                    "id": rule.id,
                    "text": rule.text,
                    "similarity": float(similarities[idx])
                })
        db.close()
        return results


memory_engine = MemoryEngine()
