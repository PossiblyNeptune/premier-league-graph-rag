# ⚽ Graph RAG Football Knowledge Chatbot : https://premier-league-graph-rag.streamlit.app/

A hybrid Retrieval-Augmented Generation (RAG) system that combines vector search, knowledge graphs, and large language models to answer questions about Premier League football history, tactics, and business.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

---

## 📖 Overview

This project implements a **Graph RAG** system specialized in Premier League football knowledge, based on two comprehensive books:
- **"The Mixer"** by Michael Cox (Tactics & Strategy)
- **"The Club"** by Robinson & Clegg (Business & History)

### Why Graph RAG?

Traditional RAG systems use only vector similarity search, which can miss related information that isn't semantically similar but is conceptually connected. **Graph RAG** solves this by:

1. **Building a Knowledge Graph** - Document chunks become nodes, relationships become weighted edges
2. **Intelligent Traversal** - Starting from vector-retrieved chunks, traverse the graph to discover connected concepts
3. **Comprehensive Context** - Accumulates rich context from multiple related sources

**Result:** More complete and nuanced answers by discovering information through relationships, not just similarity.

---

## 🌟 Key Features

- ⚡ **Hybrid Retrieval** - Combines FAISS vector search with graph traversal
- 🏈 **Football-Optimized** - Custom preprocessing for tactics, formations, and entities
- 🧠 **Local LLM** - Runs Ollama locally (no API costs, complete privacy)
- 📊 **Full Observability** - Detailed logging of retrieval decisions and graph traversal
- 🎯 **Query Classification** - Adapts retrieval strategy based on query type
- 💬 **Interactive UI** - Beautiful Streamlit interface with debug mode
- 📈 **Graph Visualizations** - Explore knowledge graph structure and connectivity

---

## 🏗️ Architecture

```
PDFs → Document Processing → Embeddings → Vector Store (FAISS)
                    ↓                            ↓
            Knowledge Graph (NetworkX)    ← Similarity Edges →
                    ↓
            Query Engine (Hybrid Retrieval)
                    ↓
            Graph Traversal (Modified Dijkstra)
                    ↓
            LLM Generation (Ollama)
                    ↓
            Streamlit UI
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Ollama (gpt-oss:120b-cloud) | Answer generation |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | 384-dim semantic vectors |
| **Vector Store** | FAISS | Fast similarity search |
| **Knowledge Graph** | NetworkX | Relationship modeling |
| **NLP** | spaCy | Named entity recognition |
| **UI** | Streamlit | Web interface |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Ollama installed ([Download](https://ollama.ai/download))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/graph-rag-football
cd graph-rag-football

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Pull Ollama model
ollama pull gpt-oss:120b-cloud

# 6. Place PDFs in data folder
mkdir data
# Copy The_Mixer.pdf and The_Club.pdf to data/

# 7. Build knowledge base (one-time setup)
python build_database.py

# 8. Launch chatbot
streamlit run chatbot.py
```

### First Run

The database build process will:
1. Load PDFs (~ 30 seconds)
2. Create embeddings (~ 2-3 minutes)
3. Build knowledge graph (~ 1-2 minutes)
4. Save to disk for future use

**Total time:** ~5 minutes

**Note:** Subsequent runs are instant (loads from saved database).

---

## 💡 Usage Examples

### Sample Queries

**Tactical Questions:**
```
- "Explain the 4-4-2 formation"
- "How did pressing tactics evolve?"
- "What is a false nine?"
```

**Manager Questions:**
```
- "What was Arsene Wenger's impact on Arsenal?"
- "Compare Ferguson's and Wenger's philosophies"
- "How did managers change English football?"
```

**Player Questions:**
```
- "Tell me about Thierry Henry's playing style"
- "Who were key players in the Invincibles season?"
```

**Business Questions:**
```
- "How did foreign investment change the Premier League?"
- "What was Roman Abramovich's impact on Chelsea?"
```

**Historical Questions:**
```
- "Why did English football tactics evolve?"
- "How did the Premier League become global?"
```

### Debug Mode

Enable in sidebar to see:
- Nodes traversed in knowledge graph
- Sources used for answer
- Retrieval confidence scores
- Graph connectivity metrics

---

## 📂 Project Structure

```
graph-rag-football/
│
├── data/                           # Input PDFs
│   ├── The_Mixer.pdf
│   └── The_Club.pdf
│
├── saved_db/                       # Persistent storage
│   ├── vector_store/              # FAISS index
│   └── knowledge_graph.pkl        # Serialized graph
│
├── graph_progress/                 # Build-time artifacts
│   ├── extraction_metadata.json
│   └── graph_statistics.json
│
├── graph_visualizations/           # Analysis dashboards
│   ├── dashboard.html
│   └── connectivity_report.json
│
├── retrieval_logs/                 # Query session logs
│   └── {session_id}.json
│
├── build_database.py              # Main: Build database
├── chatbot.py                     # Main: Streamlit UI
├── create_database.py             # Knowledge graph construction
├── create_embeddings.py           # Document processing & embeddings
├── graph_rag.py                   # Query engine & retrieval
├── football_tactics_preprocessor.py  # Domain-specific NLP
├── metadata_logger.py             # Observability & logging
├── graph_visualizer.py            # Visualization utilities
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🧠 How It Works

### 1. Document Processing

**Chunking Strategy:**
- Size: 1500 characters (~250 words)
- Overlap: 250 characters (~40 words)
- Method: Recursive splitting (paragraphs → sentences → words)

**Domain Preprocessing:**
- Normalize formations (4-4-2, 433, 4/4/2 → 4-4-2)
- Expand abbreviations (CDM → defensive midfielder)
- Extract tactical entities (pressing, false nine, etc.)
- Recognize key figures (Wenger, Ferguson, etc.)

### 2. Knowledge Graph Construction

**Node Creation:**
- Each chunk becomes a node
- Metadata: source file, page, concepts, formations

**Edge Creation:**
- Connect nodes with cosine similarity > 0.7
- Weight formula: `0.7 × similarity + 0.3 × concept_overlap`

### 3. Query Processing

**Step 1: Classification**
- Categorize query (manager, tactics, player, business, etc.)
- Adjust retrieval parameters based on type

**Step 2: Enrichment**
- Add contextual keywords
- Example: "Wenger" → "Wenger Arsene Wenger Arsenal manager..."

**Step 3: Vector Retrieval**
- FAISS finds k-nearest chunks (k=8-12 based on query type)
- Uses HuggingFace embeddings (384-dim)

**Step 4: Graph Traversal**
- Modified Dijkstra's algorithm
- Start from retrieved chunks
- Traverse weighted edges to discover related nodes
- Apply depth penalty to encourage breadth
- Stop at 20 nodes or 4000 characters

**Step 5: Answer Generation**
- Combine accumulated context
- Add type-specific guidance
- Generate with Ollama (temperature=0 for determinism)

---

## 🔬 Technical Details

### Embedding Model

**Model:** `all-MiniLM-L6-v2`
- Dimensions: 384
- Architecture: Distilled BERT (6 layers)
- Training: Contrastive learning on sentence pairs
- Performance: 0.81 Pearson correlation on STS benchmark
- Speed: ~2000 sentences/sec on CPU

### Graph Traversal Algorithm

**Modified Dijkstra:**
```python
distance = (current_priority + 1/adjusted_weight) × depth_penalty

where:
  adjusted_weight = 0.7 × edge_weight + 0.3 × concept_relevance
  depth_penalty = 1.0 + 0.05 × depth
```

**Intuition:**
- Prioritizes strong relationships (high edge weights)
- Balances semantic similarity with concept overlap
- Depth penalty prevents going too deep in one path

### Edge Weight Calculation

```python
edge_weight = α × similarity_score + β × normalized_shared_concepts

where:
  α = 0.7 (semantic similarity weight)
  β = 0.3 (concept overlap weight)
  normalized_shared_concepts = |shared| / min(|concepts₁|, |concepts₂|)
```

---

## 🛠️ Configuration

### Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud

# Chunking Parameters
CHUNK_SIZE=1500
CHUNK_OVERLAP=250

# Graph Parameters
SIMILARITY_THRESHOLD=0.7

# Retrieval Parameters
MAX_CONTEXT_LENGTH=4000
MAX_TRAVERSAL_NODES=20
```

### Adjusting Performance

**For faster responses:**
- Reduce `MAX_TRAVERSAL_NODES` to 15
- Lower `CHUNK_SIZE` to 1000

**For better quality:**
- Increase `MAX_CONTEXT_LENGTH` to 5000
- Raise `SIMILARITY_THRESHOLD` to 0.75

---

## 🐛 Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is running
ollama serve

# Verify model is available
ollama list

# Pull model if missing
ollama pull gpt-oss:120b-cloud
```

## 📈 Advanced Features

### Metadata Logging

Every query logs:
- Initial retrieval results
- Graph traversal path
- Edge weights used
- Context accumulated
- Answer confidence
- Response time

Logs saved to: `retrieval_logs/{session_id}.json`

### Graph Visualizations

After building database, explore:
- `graph_visualizations/dashboard.html` - Interactive graph explorer
- `graph_visualizations/connectivity_report.json` - Graph metrics

---

## 🙏 Acknowledgments

### Source Materials
- **"The Mixer"** by Michael Cox - Tactical evolution of English football
- **"The Club"** by Joshua Robinson & Jonathan Clegg - Business history of Premier League

### Technologies
- [Ollama](https://ollama.ai) - Local LLM inference
- [HuggingFace](https://huggingface.co) - Sentence transformers
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [NetworkX](https://networkx.org) - Graph data structures
- [spaCy](https://spacy.io) - Named entity recognition
- [Streamlit](https://streamlit.io) - Web UI framework
- [LangChain](https://langchain.com) - LLM orchestration

