import pandas as pd
import streamlit as st

from retrieval_models.tfidf_retrieval import TFIDFRetrieval
from retrieval_models.semantic_retrieval import SemanticRetrieval


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Information Retrieval System",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLES
# ---------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    padding:40px;
    border-radius:12px;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
    margin-bottom:30px;
}

.hero h1{
    font-size:48px;
}

.hero p{
    font-size:18px;
    opacity:0.9;
}

.result-card{
    background:white;
    padding:20px;
    border-radius:12px;
    margin-bottom:20px;
    border:1px solid #eaeaea;
    transition:0.2s;
}

.result-card:hover{
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
}

.search-container{
    padding:30px;
    border-radius:10px;
    background:#f7f9fc;
    margin-bottom:30px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:10px;
    text-align:center;
    border:1px solid #eee;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

docs = pd.read_csv("data/ir_documents.csv")

tfidf_model = TFIDFRetrieval(docs)
semantic_model = SemanticRetrieval(docs)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("System Dashboard")

st.sidebar.metric("Documents Indexed", len(docs))
st.sidebar.metric("Models Available", "2")
st.sidebar.metric("System Status", "Active")

st.sidebar.markdown("---")
st.sidebar.info("AI Retrieval using TF-IDF and Semantic Search")


# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🔎 Intelligent Information Retrieval System</h1>
<p>
An AI-powered search engine for retrieving relevant knowledge
from the university information repository using advanced
machine learning retrieval techniques.
</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SEARCH PANEL
# ---------------------------------------------------

st.markdown('<div class="search-container">', unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    query = st.text_input("Enter your search query")

with col2:
    model_choice = st.selectbox(
        "Retrieval Model",
        ["TF-IDF", "Semantic AI"]
    )

search = st.button("🔍 Search")

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

if search and query:

    if model_choice == "TF-IDF":
        results = tfidf_model.search(query)
    else:
        results = semantic_model.search(query)

    st.subheader("Top Search Results")

    for rank, doc_id in enumerate(results, start=1):

        doc = docs[docs.doc_id == doc_id].iloc[0]

        st.markdown(f"""
        <div class="result-card">

        <h3>#{rank} {doc['title']}</h3>

        <p>{doc['content']}</p>

        </div>
        """, unsafe_allow_html=True)
        
st.subheader("Evaluation Results")

results = pd.read_csv("evaluation_results.csv")

st.dataframe(results)
