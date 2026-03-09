from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetrieval:

    def __init__(self, documents):

        self.docs = documents['content'].tolist()

        self.vectorizer = TfidfVectorizer()

        self.doc_vectors = self.vectorizer.fit_transform(self.docs)

        self.doc_ids = documents['doc_id'].tolist()

    def search(self, query, top_k=5):

        query_vec = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = [self.doc_ids[i] for i in ranked_indices]

        return results