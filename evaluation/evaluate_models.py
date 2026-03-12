import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from retrieval_models.tfidf_retrieval import TFIDFRetrieval
from retrieval_models.semantic_retrieval import SemanticRetrieval


docs = pd.read_csv("data/ir_documents.csv")
queries = pd.read_csv("data/ir_queries.csv")
relevance = pd.read_csv("data/ir_relevance.csv")


tfidf_model = TFIDFRetrieval(docs)
semantic_model = SemanticRetrieval(docs)


results = []


for _, row in queries.iterrows():

    query_id = row["query_id"]
    query = row["query"]

    relevant_docs = relevance[relevance.query_id == query_id]["doc_id"].tolist()

    tfidf_results = tfidf_model.search(query)
    semantic_results = semantic_model.search(query)


    # Create binary vectors
    y_true = [1 if doc in relevant_docs else 0 for doc in docs.doc_id]

    tfidf_pred = [1 if doc in tfidf_results else 0 for doc in docs.doc_id]
    semantic_pred = [1 if doc in semantic_results else 0 for doc in docs.doc_id]


    tfidf_precision = precision_score(y_true, tfidf_pred)
    tfidf_recall = recall_score(y_true, tfidf_pred)
    tfidf_f1 = f1_score(y_true, tfidf_pred)


    semantic_precision = precision_score(y_true, semantic_pred)
    semantic_recall = recall_score(y_true, semantic_pred)
    semantic_f1 = f1_score(y_true, semantic_pred)


    results.append({
        "Query": query,
        "TFIDF Precision": tfidf_precision,
        "TFIDF Recall": tfidf_recall,
        "TFIDF F1": tfidf_f1,
        "Semantic Precision": semantic_precision,
        "Semantic Recall": semantic_recall,
        "Semantic F1": semantic_f1
    })


df = pd.DataFrame(results)

print(df)

df.to_csv("evaluation_results.csv", index=False)