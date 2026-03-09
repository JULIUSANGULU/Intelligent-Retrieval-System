import pandas as pd

from retrieval_models.boolean_retrieval import BooleanRetrieval
from retrieval_models.tfidf_retrieval import TFIDFRetrieval
from retrieval_models.semantic_retrieval import SemanticRetrieval

from evaluation.evaluation_metrics import precision_at_k, recall


# ==============================
# Load Dataset
# ==============================

docs = pd.read_csv("data/ir_documents.csv")
queries = pd.read_csv("data/ir_queries.csv")
relevance = pd.read_csv("data/ir_relevance.csv")


# ==============================
# Initialize Retrieval Models
# ==============================

boolean_model = BooleanRetrieval(docs)
tfidf_model = TFIDFRetrieval(docs)
semantic_model = SemanticRetrieval(docs)


# ==============================
# Store Results
# ==============================

results = []


# ==============================
# Run Experiment
# ==============================

for index, row in queries.iterrows():

    query = row["query"]
    qid = row["query_id"]

    relevant_docs = relevance[relevance.query_id == qid]["doc_id"].tolist()

    # Retrieve documents
    boolean_results = boolean_model.search(query)
    tfidf_results = tfidf_model.search(query)
    semantic_results = semantic_model.search(query)

    # Compute Metrics
    boolean_precision = precision_at_k(boolean_results, relevant_docs, 5)
    boolean_recall = recall(boolean_results, relevant_docs)

    tfidf_precision = precision_at_k(tfidf_results, relevant_docs, 5)
    tfidf_recall = recall(tfidf_results, relevant_docs)

    semantic_precision = precision_at_k(semantic_results, relevant_docs, 5)
    semantic_recall = recall(semantic_results, relevant_docs)

    # Store results
    results.append({

        "query": query,

        "boolean_precision": boolean_precision,
        "boolean_recall": boolean_recall,

        "tfidf_precision": tfidf_precision,
        "tfidf_recall": tfidf_recall,

        "semantic_precision": semantic_precision,
        "semantic_recall": semantic_recall

    })


# ==============================
# Save Results
# ==============================

results_df = pd.DataFrame(results)

results_df.to_csv("results/experiment_results.csv", index=False)


# ==============================
# Print Average Performance
# ==============================

print("\n===== Average System Performance =====\n")

print("Boolean Precision:", results_df["boolean_precision"].mean())
print("Boolean Recall:", results_df["boolean_recall"].mean())

print("\nTF-IDF Precision:", results_df["tfidf_precision"].mean())
print("TF-IDF Recall:", results_df["tfidf_recall"].mean())

print("\nSemantic Precision:", results_df["semantic_precision"].mean())
print("Semantic Recall:", results_df["semantic_recall"].mean())