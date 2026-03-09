import streamlit as st
import pandas as pd

from retrieval_models.tfidf_retrieval import TFIDFRetrieval
from retrieval_models.semantic_retrieval import SemanticRetrieval

# Load documents
docs = pd.read_csv("data/ir_documents.csv")

# Initialize models
tfidf_model = TFIDFRetrieval(docs)
semantic_model = SemanticRetrieval(docs)

st.title("Intelligent Information Retrieval System")

st.write("Search the university knowledge base using AI-powered retrieval.")

query = st.text_input("Enter your search query")

model_choice = st.selectbox(
    "Select Retrieval Model",
    ["TF-IDF", "Semantic AI"]
)

if st.button("Search"):

    if model_choice == "TF-IDF":
        results = tfidf_model.search(query)

    else:
        results = semantic_model.search(query)

    st.subheader("Top Results")

    for doc_id in results:

        doc = docs[docs.doc_id == doc_id].iloc[0]

        st.write("###", doc["title"])
        st.write(doc["content"])
        st.write("---")