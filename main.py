import pandas as pd

from retrieval_models.boolean_retrieval import BooleanRetrieval
from retrieval_models.tfidf_retrieval import TFIDFRetrieval
from retrieval_models.semantic_retrieval import SemanticRetrieval

docs = pd.read_csv("data/ir_documents.csv")
queries = pd.read_csv("data/ir_queries.csv")
relevance = pd.read_csv("data/ir_relevance.csv")

boolean_model = BooleanRetrieval(docs)
tfidf_model = TFIDFRetrieval(docs)
semantic_model = SemanticRetrieval(docs)

for index, row in queries.iterrows():

    query = row['query']
    qid = row['query_id']

    relevant_docs = relevance[relevance.query_id == qid]['doc_id'].tolist()

    boolean_results = boolean_model.search(query)
    tfidf_results = tfidf_model.search(query)
    semantic_results = semantic_model.search(query)

    print("Query:", query)
    print("Boolean:", boolean_results[:5])
    print("TF-IDF:", tfidf_results[:5])
    print("Semantic:", semantic_results[:5])