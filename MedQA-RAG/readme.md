# 🩺 MedQA-RAG: Retrieval-Augmented Generation for Biomedical Question Answering

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline on the **MedQA (USMLE-style)** biomedical dataset.  
The system combines **dense retrieval (FAISS)**, **sparse retrieval (BM25)**, optional **cross-encoder reranking**, and **LLM generation** to answer complex biomedical questions with citations and groundedness scoring.

---

## 📑 Project Overview

- **Dataset**: [MedQA (USMLE version)](https://github.com/jind11/MedQA) – preprocessed into JSONL format (`qa.jsonl`).
- **Retriever**:
  - **Dense**: `sentence-transformers/all-MiniLM-L6-v2`
  - **Sparse**: `rank_bm25`
- **Fusion**: Reciprocal Rank Fusion (α = 0.6) between dense and sparse retrievers.
- **Reranker (optional)**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generator**: `google/flan-t5-base` (can swap for `flan-t5-large` for stronger results).
- **Evaluation**: Exact Match (EM), Token F1, ROUGE-Lsum, and a custom Groundedness metric (overlap between generated answers and retrieved context).

---

## ⚙️ Pipeline Architecture

Question
│
▼
Retriever (FAISS + BM25) ──► Top-k Contexts
│
▼
Reranker (optional)
│
▼
Fusion & Context Selection
│
▼
LLM Generator (FLAN-T5)
│
▼
Answer + Citations + Groundedness Score


---

## 📊 Results

Evaluation was run on **100 samples** from MedQA:

| Metric        | Score |
|---------------|-------|
| Exact Match   | 0.12  |
| Token F1      | 0.28  |
| ROUGE-Lsum    | 0.25  |
| Groundedness  | 0.45  |

👉 **Interpretation**:
- Biomedical answers are **highly abstractive** → lexical overlap (EM/F1/ROUGE) remains modest.  
- **Groundedness (0.45)** shows the model is partly anchored to retrieved evidence.  
- With stronger LLMs (e.g. `flan-t5-large` or GPT-4-mini), overlap metrics improve significantly.

---

## 🚀 Getting Started

### 1. Clone the repo

git clone https://github.com/your-username/MedQA-RAG.git
cd MedQA-RAG

2. Install dependencies
pip install -r requirements.txt

3. Preprocess data
python scripts/preprocess_data.py

4. Build retrieval indexes
python scripts/build_indexes.py

5. Run evaluation
python scripts/eval_generation.py \
  --limit 100 \
  --k 12 \
  --alpha 0.6 \
  --max-new-tokens 200 \
  --temperature 0.1 \
  --extractive-first


Results are saved to:

eval/generation_results.csv

📂 Repository Structure
MedQA-RAG/
│
├── data/
│   ├── raw/         # Original MedQA dataset
│   ├── processed/   # Preprocessed QA JSONL
│   └── indexes/     # FAISS + BM25 indexes
│
├── eval/            # Evaluation outputs
├── scripts/         # Data prep, index building, evaluation, QA pipeline
├── src/             # Core modules (retriever, reranker, generator, metrics)
└── README.md

🧪 Example Usage

Ask a custom question:

python scripts/ask.py \
  --q "What are the adverse effects of aspirin?" \
  --model google/flan-t5-base \
  --temperature 0.0


Example output:

Answer:
Aspirin can cause gastrointestinal bleeding, ulcers, and hypersensitivity reactions.

Groundedness: 0.62
Citations:
 - [PMID:12345678] Aspirin and GI complications

 - [PMID:98765432] Hypersensitivity to salicylates
