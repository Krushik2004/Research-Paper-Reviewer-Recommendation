import os
import fitz
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

def extract_first_n_words(pdf_path, n_words=2000):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        if len(doc) >= 100:
            return ""
        word_count = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            words = page_text.split()
            if not words:
                continue
            if word_count + len(words) > n_words:
                remaining = n_words - word_count
                text += " " + " ".join(words[:remaining])
                break
            else:
                text += " " + page_text
                word_count += len(words)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def build_paper_level_csv(dataset_path, output_csv="papers.csv"):
    """Extract author name, file name, and text (first 3 pages)"""
    data = []
    for author in tqdm(os.listdir(dataset_path), desc="Extracting papers"):
        author_dir = os.path.join(dataset_path, author)
        if not os.path.isdir(author_dir):
            continue
        for file in os.listdir(author_dir):
            if not file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(author_dir, file)
            text = extract_first_n_words(pdf_path)
            if len(text) > 100:
                data.append({"Author": author, "File": file, "Text": text})
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["Author", "Text"], inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved paper-level data to {output_csv}")
    return df

def compute_embeddings(model, csv, embeddings_path):
    """Compute embeddings for each author's combined text"""
    df = pd.read_csv(csv)

    print(f"Generating embeddings for {len(df)} items...")
    embeddings = model.encode(df["Text"].tolist(), show_progress_bar=True)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    return embeddings

def build_tfidf_vectors(paper_df):
    """Fit TF-IDF vectorizer on author texts"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=30000)
    tfidf_matrix = vectorizer.fit_transform(paper_df["Text"].tolist())
    print(f"Built TF-IDF matrix for {len(paper_df)} authors, {tfidf_matrix.shape[1]} features.")
    return vectorizer, tfidf_matrix

def recommend_top_k_authors(pdf_path, k, paper_df, paper_embeddings, model, vectorizer, tfidf_matrix, nmf_model, w_emb, w_vec):
    if not(0 <= w_emb <= 1 and 0 <= w_vec <= 1 and w_emb + w_vec <= 1):
        print(f"Weights do not satisfy condition: w_emb - {w_emb} + w_vec - {w_vec} = {w_emb + w_vec}")
        return pd.DataFrame()
    
    w_topic = 1 - (w_emb + w_vec)
    
    new_text = extract_first_n_words(pdf_path)

    if not new_text:
        print("Could not extract text from input PDF.")
        return pd.DataFrame()

    new_emb = model.encode([new_text])
    emb_sims = cosine_similarity(new_emb, paper_embeddings)[0]

    new_vec = vectorizer.transform([new_text])
    vec_sims = cosine_similarity(new_vec, tfidf_matrix)[0]

    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_
    new_topic_vec = nmf_model.transform(new_vec)
    topic_sims = cosine_similarity(new_topic_vec, W)[0]

    sims = w_emb * emb_sims + w_vec * vec_sims + w_topic * topic_sims

    identical_indices = np.where(np.isclose(emb_sims, 1.0, atol=1e-6))[0]
    excluded_authors = set()
    if len(identical_indices) > 0:
        excluded_authors = set(paper_df.iloc[identical_indices]["Author"].tolist())
        print(f"Excluding authors with identical papers: {', '.join(excluded_authors)}")

    filtered_indices = [
        i for i in range(len(sims)) if paper_df.iloc[i]["Author"] not in excluded_authors
    ]
    filtered_sims = sims[filtered_indices]

    percentile_threshold = np.percentile(filtered_sims, 95)
    print(f"95th percentile similarity threshold: {percentile_threshold:.4f}")

    author_scores = {}
    for i, row in paper_df.iterrows():
        author = row["Author"]
        if author in excluded_authors:
            continue
        sim = sims[i]
        if sim >=  percentile_threshold and sim > 0:
            author_scores.setdefault(author, []).append(sim)

    final_scores = []
    for author, sim_list in author_scores.items():
        num_papers = len(sim_list)
        avg_sim = np.mean(sim_list)
        weight = math.log2(num_papers + 1)
        score = avg_sim * weight
        final_scores.append({
            "Author": author,
            "Average Similarity (avg_sim)": avg_sim,
            "Relevant Papers (n)": num_papers,
            "Final Score = avg_sim * math.log2(n + 1)": score
        })

    result_df = pd.DataFrame(final_scores)
    if result_df.empty:
        print("No authors found with similarity > 0.")
        return result_df

    result_df = result_df.sort_values(by="Final Score = avg_sim * math.log2(n + 1)", ascending=False).head(k)

    print("\nTop-k Recommended Authors:")
    print(result_df)
    return result_df