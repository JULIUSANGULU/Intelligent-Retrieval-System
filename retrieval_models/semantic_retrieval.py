from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRetrieval:

    def __init__(self, documents):

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.doc_texts = documents['content'].tolist()

        self.doc_ids = documents['doc_id'].tolist()

        self.doc_embeddings = self.model.encode(self.doc_texts)

    def search(self, query, top_k=5):

        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = [self.doc_ids[i] for i in ranked_indices]

        return results