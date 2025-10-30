# 📄 Research Paper Reviewer Recommendation System

This project is an **AI-powered Research Paper Reviewer Recommendation System** that automatically recommends the most suitable reviewers for a given research paper.  
It leverages **Natural Language Processing (NLP)** and **text similarity techniques** to analyze and compare the textual content of research papers.

---

## 🎯 **Objective**

The goal of this project is to assist in **automating the reviewer assignment process** by analyzing a newly submitted research paper and recommending potential reviewers whose past work is most similar to the submitted paper’s content.

---

## 🧠 **Approach Overview**

The system extracts textual content from PDFs and computes similarities using **three complementary techniques**:

| Similarity Type | Technique Used | Description |
|-----------------|----------------|--------------|
| **Semantic Similarity** | SentenceTransformer (`all-mpnet-base-v2`) | Captures contextual meaning of text using transformer-based embeddings. |
| **Text-based Similarity** | TF-IDF Vectorization | Measures term-level similarity based on word frequency. |
| **Topic Similarity** | Non-negative Matrix Factorization (NMF) | Compares latent topic distributions between research papers. |

These measures are combined into a **weighted similarity score** to identify the most relevant reviewers.

---

## ⚙️ **Reviewer Scoring Mechanism**

Each uploaded research paper is compared with existing papers in the dataset using the three similarity measures mentioned above.

### Weight Distribution:
- **Semantic Similarity (w₁ = 0.6)**  
- **TF-IDF Similarity (w₂ = 0.25)**  
- **Topic Similarity (w₃ = 0.15)**  

The overall author score is computed as:

$$
\mathrm{Final\ Score} = \mathrm{avg\underline similarity} \times \log_2\bigl(\mathrm{n} + 1\bigr)
$$

Where:
- **avg_similarity** = average similarity of papers with similarity ≥ 95th percentile  
- **n** = number of papers with similarity > 0  

This helps balance both **relevance** and **expertise breadth** of the reviewers.

---
