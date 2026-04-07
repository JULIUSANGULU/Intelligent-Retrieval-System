import pandas as pd
import streamlit as st

from retrieval_models.bm25_retrieval import BM25Retrieval
from retrieval_models.sbert_retrieval import SBERTRetrieval


# PAGE CONFIG
st.set_page_config(
    page_title="AI Information Retrieval System",
    layout="wide"
)

# CUSTOM STYLES
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


# LOAD DATA
docs = pd.read_csv("data/ir_documents.csv")

bm25_model = BM25Retrieval(docs)
sbert_model = SBERTRetrieval(docs)


# SIDEBAR
st.sidebar.title("System Dashboard")

st.sidebar.metric("Documents Indexed", len(docs))
st.sidebar.metric("Models Available", "2")
st.sidebar.metric("System Status", "Active")

st.sidebar.markdown("---")
st.sidebar.info("Baseline: BM25 | Intelligent Model: SBERT")


# HERO SECTION
st.markdown("""
<div class="hero">
<h1> Intelligent Information Retrieval System</h1>
<p>
A hybrid AI-powered search system designed to compare traditional 
and intelligent retrieval techniques for improved information access 
within a university knowledge base.
</p>
</div>
""", unsafe_allow_html=True)


# SEARCH PANEL
st.markdown('<div class="search-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3,1,1])

with col1:
    query = st.text_input("Enter your search query")

with col2:
    model_choice = st.selectbox(
        "Retrieval Model",
        ["BM25 (Baseline)", "SBERT (Semantic AI)"]
    )

# 🔥 NEW (Supervisor requirement)
with col3:
    query_type = st.selectbox(
        "Query Type",
        ["Normal", "Short Query", "Synonym", "Paraphrased"]
    )

search = st.button("🔍 Search")

st.markdown("</div>", unsafe_allow_html=True)


# RESULTS
if search and query:

    st.info(f"Query Type: {query_type}")

    if model_choice == "BM25 (Baseline)":
        results = bm25_model.search(query)
    else:
        results = sbert_model.search(query)

    st.subheader(f"Top {len(results)} Search Results")

    for rank, doc_id in enumerate(results, start=1):

        doc = docs[docs.doc_id == doc_id].iloc[0]

        st.markdown(f"""
        <div class="result-card">

        <h3>#{rank} {doc['title']}</h3>

        <p>{doc['content']}</p>

        </div>
        """, unsafe_allow_html=True)


# EVALUATION RESULTS
st.subheader("Evaluation Results")

try:
    results_df = pd.read_csv("evaluation_results.csv")
    st.dataframe(results_df)
except:
    st.warning("Run evaluation script to generate results.")