
# ⚽ Football Graph-RAG Chatbot
*A Hybrid Retrieval-Augmented Generation System Built on Football Tactics Literature*

This project implements a **Graph-Based Retrieval-Augmented Generation (Graph RAG)** chatbot trained on two football analytics books:

- **The Mixer** (Michael Cox)
- **The Club** (Joshua Robinson & Jonathan Clegg)

Instead of relying purely on vector similarity search, the system builds a **domain-aware knowledge graph** of tactical concepts, clubs, managers, players, and strategic relationships — enabling **context-rich, accurate answers** in football tactical discussions.

---

## 🧠 System Overview

The project uses a **two-stage retrieval pipeline**:

| Stage | Component | Purpose |
|------|----------|---------|
| **1. Semantic Retrieval** | FAISS Vector Store | Finds relevant text chunks by **meaning**, not exact wording |
| **2. Graph Traversal** | NetworkX Knowledge Graph | Expands retrieved contexts using **tactical + conceptual relations** |

This hybrid approach ensures:
- More **context depth** than raw embeddings
- Better **tactical reasoning**
- More **consistent and grounded** answers

---

## 🏗 Architecture Flow

PDF Books → Preprocessing → Chunking → Embeddings → FAISS Store  
                  ↘  
            Knowledge Graph (NetworkX)  

Query → Vector Retrieval (FAISS) → Graph Expansion → Answer Generation (Ollama)

---

## 📦 Tech Stack & Purpose

| Library | Role |
|--------|------|
| sentence-transformers | Generates vector embeddings |
| FAISS | High-speed similarity search |
| NetworkX | Graph construction & traversal |
| pandas/numpy | Data processing |
| plotly/matplotlib | Visualization |
| python-docx/reportlab | Report generation |
| Ollama | Local LLM execution |

---

## 📂 File Structure

project/
├── build_database.py
├── chatbot.py
├── create_database.py
├── create_embeddings.py
├── football_tactics_preprocessor.py
├── graph_rag.py
├── graph_visualizer.py
├── metadata_logger.py
└── data/ (source pdfs)

---

## 🔧 Install

```bash
pip install -r requirements.txt
```
Install Ollama manually: https://ollama.ai/download

---

## 🚀 Run

```bash
python build_database.py
streamlit run chatbot.py
```

---

## 🎯 Example Query

> How did Sir Alex Ferguson adapt pressing at United during the early Premier League years?

---

## 📊 Visualization

```bash
python graph_visualizer.py
```
Outputs HTML dashboards inside `graph_visualizations/`.

---

## 🛠 Logging

Session logs stored in `retrieval_logs/`.

---

## 📈 Possible Enhancements

- Neo4j backend for large graphs
- Fine-tuned tactical embedding models
- Streamlit UI

---

## ✅ Summary

This system combines **semantic search + structured tactical relationships**, producing more meaningful football tactical answers than standard RAG.