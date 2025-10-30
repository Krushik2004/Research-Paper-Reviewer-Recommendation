import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from recommender import (
    extract_first_n_words,
    recommend_top_k_authors,
    build_tfidf_vectors
)

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="Research Paper Reviewer Recommender",
    page_icon="📄",
    layout="centered",
)

# -------------------------------
# Custom styling
# -------------------------------
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #2E8BCA;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #E0E0E0;
            margin-bottom: 25px;
        }
        .upload-section {
            background-color: transparent;  /* remove white box */
            padding: 15px;
            border-radius: 15px;
            text-align: center;
            box-shadow: none;
        }
        .result-table {
            border-radius: 10px;
            background-color: #f9f9f9;
            padding: 10px;
        }
        /* 🟢 Clean, minimal button styling */
        div[data-testid="stButton"] button {
            width: 100% !important;
            height: 38px !important;
            background-color: #3A6EA5 !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            transition: background-color 0.2s ease-in-out !important;

            /* ✅ subtle alignment fix */
            padding-top: 2px !important;
            padding-bottom: 2px !important;
            line-height: 1.2 !important;
            margin-top: 11px !important;   /* ⬅️ slightly increased from 4px to 6px */
        }


    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>📄 Research Paper Reviewer Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a research paper PDF and get top recommended reviewers based on multiple similarity measures.</p>", unsafe_allow_html=True)

# -------------------------------
# Upload section
# -------------------------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload your research paper (PDF)", type=["pdf"])

col1, col2 = st.columns([4, 1])
with col1:
    k = st.selectbox("Select number of reviewers to recommend (k)", [1, 2, 3, 4, 5], index=4)
with col2:
    with st.container():
        st.markdown("<div class='small-btn'>", unsafe_allow_html=True)
        run_button = st.button("🔍 Find Reviewers", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)
if "ran_once" not in st.session_state:   # 🟢 Track whether user has pressed the button
    st.session_state.ran_once = False
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Fixed parameters
# -------------------------------
w_emb = 0.6
w_vec = 0.25
nmf_topics = 40

# -------------------------------
# Load precomputed data and models
# -------------------------------
@st.cache_resource
def load_resources():
    papers_csv = "papers.csv"
    embeddings_path = "paper_embeddings.npy"

    paper_df = pd.read_csv(papers_csv)
    paper_embeddings = np.load(embeddings_path)
    model = SentenceTransformer("all-mpnet-base-v2")

    vectorizer, tfidf_matrix = build_tfidf_vectors(paper_df)
    nmf_model = NMF(n_components=nmf_topics, max_iter=300, random_state=42)

    return paper_df, paper_embeddings, model, vectorizer, tfidf_matrix, nmf_model

paper_df, paper_embeddings, model, vectorizer, tfidf_matrix, nmf_model = load_resources()


if "result_df" not in st.session_state:
    st.session_state.result_df = pd.DataFrame()

# -------------------------------
# Process uploaded PDF
# -------------------------------
if run_button:
    st.session_state.ran_once = True
    if uploaded_file is not None:
        # st.write("### 🧩 Processing uploaded paper...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            test_pdf = tmp_file.name

        with st.spinner("🔍 Analyzing and computing similarities..."):
            result_df = recommend_top_k_authors(
                test_pdf, k, paper_df, paper_embeddings, model,
                vectorizer, tfidf_matrix, nmf_model, w_emb, w_vec
            )

        if not result_df.empty:
            # Clean up dataframe display
            result_df = result_df.reset_index(drop=True)
            result_df.index = np.arange(1, len(result_df) + 1)  # ranks 1..k
            result_df.index.name = "Rank"

            st.session_state.result_df = result_df
            st.success(f"✅ Top {k} Recommended Reviewers Found!")
        else:
            st.session_state.result_df = pd.DataFrame()
            st.warning("No suitable reviewers found. Try another paper or dataset.")
    else:
        st.session_state.result_df = pd.DataFrame()
        st.warning("Please upload a research paper PDF before running the recommendation.")

if not st.session_state.result_df.empty:
    st.markdown("<div class='result-table'>", unsafe_allow_html=True)
    st.dataframe(st.session_state.result_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
elif not st.session_state.ran_once:
    # 🟢 Show only initial instruction, not when button clicked
    st.info("Please upload a PDF and click **Find Reviewers** to start.")
