## Team Members:
- Krushik Teja Thati — Roll No: SE2UARI078
- Madala Venkata Bhargav — Roll No: SE22UARI086
- Neelarapu Sai Shivan Raj — Roll No: SE22UARI106
- Merugu MouryaAdithya — Roll No: SE22UARI095

# 📄 Research Paper Reviewer Recommendation System

This project is an **AI-powered Research Paper Reviewer Recommendation System** that automatically recommends the most suitable reviewers for a given research paper.  
It leverages **Natural Language Processing (NLP)** and **text similarity techniques** to analyze and compare the textual content of research papers.

---

## 🎯 **Objective**

The goal of this project is to assist in **automating the reviewer assignment process** by analyzing a newly submitted research paper and recommending potential reviewers whose past work is most similar to the submitted paper’s content.

---

**The Deployed Website link hosted on  streamlit**: https://research-paper-reviewer-recommendation-jqyyvdqdnqrnvgnmcn7v7y.streamlit.app

## 🧠 **Approach Overview**
### Dataset and PDF Processing:
- The dataset was provided by the course instructor. Dataset is available here: https://mahindraecolecentrale-my.sharepoint.com/:f:/g/personal/sanatan_sukhija_mahindrauniversity_edu_in/Es7ZAw7fvnlIqX0DmkCPiYwBQ5ueGz33Hq-NALrQKhgzww
- It consists of multiple folders, each named after an author, containing their published research papers in PDF format.
- Since research papers vary widely in structure and formatting, we used a PDF parsing pipeline to extract text uniformly.
- From each PDF, the first 2000 words were extracted using PyMuPDF (fitz), ensuring that the most representative parts (title, abstract, introduction, and early methodology) were captured. The names of authors were extracted from the directory name containing the research papers due to research papers having various formats, which made extracting particular details difficult.
- Research papers that contained too few words or excessive pages (≥100) were excluded to maintain data quality. For particular author(directory) duplicate research papers were discarded.

### Feature Representation:

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
\mathrm{Final\ Score} = \mathrm{avg\underline{ }similarity} \times \log_2\bigl(\mathrm{n} + 1\bigr)
$$

Where:
- **avg_similarity** = average similarity of papers with similarity ≥ 95th percentile  
- **n** = number of papers with similarity > 0

## 🚀 Step-by-Step Setup Instructions

### 🪜 Step 1: Clone the Repository
```bash
git clone https://github.com/Krushik2004/Research-Paper-Reviewer-Recommendation.git
cd research-paper-reviewer-recommender
```

This helps balance both **relevance** and **expertise breadth** of the reviewers.

---

### 🪜 Step 2: Set Up Python Environment
It’s recommended to create a virtual environment before installation.
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 🪜 Step 3: Install Required Dependencies
All required libraries are listed in requirements.txt.
```nginx
pip install -r requirements.txt
Contents of requirements.txt:
streamlit
pandas
numpy
fitz
sentence-transformers
scikit-learn
tqdm
```

### 🪜 Step 4: Run the Streamlit Application
Launch the web application.
```python
streamlit run app.py
```
This opens a local server — typically at:
```arduino
http://localhost:8501
```

### 🪜 Step 5: Use the Web App
- Upload a research paper (PDF).
- Select the number of reviewers (k) to recommend.
- Click “Find Reviewers”.
- The system computes similarity scores and displays a ranked list of reviewers.

### 💻 Deploying to Streamlit Cloud
You can host this app online using Streamlit Cloud.
Steps:
- Push your repository to GitHub.
- Go to Streamlit Cloud.
- Connect your GitHub account and select this repository.
- In “Advanced Settings”, set the start command as:
```python
streamlit run app.py
```
- Click Deploy 🎉
