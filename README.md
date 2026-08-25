# 📚 arXiv RAG Research Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system for asking questions about academic research papers from **arXiv**.

The project combines **semantic chunking, Gemini embeddings, ChromaDB, query expansion, vector retrieval, LLM-based reranking, and grounded answer generation** to retrieve relevant research papers and generate answers with supporting sources.

It now includes a lightweight **FastAPI backend** and **React/Vite frontend** for interacting with the RAG pipeline through a web interface.

---

## ✨ Features

* 📄 Extracts and processes research papers from PDFs
* 🧩 **LLM-based semantic chunking** to preserve meaningful sections
* 🧠 **Gemini embeddings** for semantic search
* 🗃️ **ChromaDB** for persistent vector storage
* 🔀 **Query expansion** to improve retrieval coverage
* 🔍 Multi-query vector retrieval
* 🎯 **LLM-based reranking** of retrieved chunks
* 🤖 Grounded answer generation using retrieved context
* 📚 Returns the actual retrieved source papers
* 📊 Evaluation using **Retrieval Accuracy, MRR, NDCG@4, and Fact Recall**
* ⚡ **FastAPI REST API** for serving the RAG pipeline
* ⚛️ **React + Vite frontend** for asking research questions
* 🎨 Minimal frontend built with plain CSS
* 🔄 Loading and error handling in the frontend
* 🖥️ Original Gradio interface is also included

---

## 🏗️ Architecture

The frontend does **not** implement a second retrieval pipeline. It communicates with the existing RAG system through the FastAPI `/ask` endpoint.

---

## 🛠️ Tech Stack

### Backend & RAG

**Language:** Python

**LLM:** Google Gemini 2.5 Flash

**Embeddings:** Gemini Embeddings

**RAG / Vector Store:** LangChain, ChromaDB

**Document Processing:** PyPDF

**API:** FastAPI, Uvicorn

### Frontend

**Framework:** React

**Build Tool:** Vite

**Styling:** Plain CSS

### Evaluation

**Format:** JSON / JSONL

**Metrics:**

* Retrieval Accuracy
* Mean Reciprocal Rank (MRR)
* NDCG@4
* Fact Recall

---

## 📁 Project Structure

```text
arXiv_RAG_project/
│
├── app.py                  # Original Gradio application
├── server.py               # FastAPI backend
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── index.css       # Frontend styling
│   │   └── main.jsx        # React entry point
│   │
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── src/
│   ├── ingest.py           # PDF processing & semantic chunking
│   ├── embed.py            # Embedding generation & ChromaDB
│   ├── answer.py           # Retrieval, reranking & answer generation
│   ├── eval.py             # RAG evaluation
│   ├── benchmark.py        # Benchmarking utilities
│   └── check_models.py     # Gemini model utility
│
├── data/
│   ├── raw/                # Research papers
│   ├── chunks.json         # Processed chunks
│   └── chroma_db/          # Persistent vector database
│
├── evals/
│   └── eval_set.jsonl      # Evaluation dataset
│
├── .gitignore
├── README.md
└── ...
```

> `node_modules/`, Python virtual environments, `.env`, and Python cache files are excluded using `.gitignore`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vansh-glitch1505/arXiv_RAG_project.git
cd arXiv_RAG_project
```

### 2. Install Python dependencies

```bash
pip install arxiv requests pypdf python-dotenv langchain-chroma langchain-google-genai langchain-core gradio google-genai fastapi uvicorn
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit your `.env` file to GitHub.

---

## ⚙️ Build the RAG Pipeline

### Process the research papers

```bash
python src/ingest.py
```

### Generate embeddings and build the vector database

```bash
python src/embed.py
```

---

# 🖥️ Run the Web Application

The application consists of two services:

```text
Frontend → http://localhost:5173
Backend  → http://localhost:8000
```

Both need to be running.

### 1. Start the FastAPI backend

From the project root:

```bash
python server.py
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

### 2. Start the React frontend

Open a second terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

---

## 🔌 API

### `POST /ask`

The frontend sends a research question to the FastAPI backend.

#### Request

```json
{
  "question": "How large was the SwiLTra-Bench test set used for evaluation?"
}
```

#### Response

```json
{
  "answer": "The SwiLTra-Bench test set used for evaluation consists of roughly 18.1k sentence pairs.",
  "sources": [
    "2607.19226v1"
  ]
}
```

The `sources` field is generated from the **actual reranked documents returned by the existing RAG pipeline** rather than using hardcoded sources.

---

## 🔄 RAG Pipeline

The system follows the following workflow:

```text
Research Papers
      ↓
PDF Extraction
      ↓
Semantic Chunking
      ↓
Gemini Embeddings
      ↓
ChromaDB
      ↓
Query Expansion
      ↓
Multi-Query Retrieval
      ↓
LLM Reranking
      ↓
Relevant Context
      ↓
Answer Generation
      ↓
Answer + Source Papers
```

The FastAPI layer simply exposes this existing pipeline through a REST endpoint.

---

## 📊 Evaluation

The project includes an evaluation dataset and measures the performance of the RAG system using:

### Retrieval Accuracy

Measures whether the relevant paper is successfully retrieved.

### Mean Reciprocal Rank (MRR)

Measures how highly the relevant paper appears in the ranked retrieval results.

### NDCG@4

Measures the quality of the top four retrieved results while considering their ranking.

### Fact Recall

Measures whether the generated answer contains the expected facts from the retrieved research papers.

The evaluation dataset contains research questions with expected papers and facts for measuring retrieval and answer quality.

---

## 🧪 Example Query

A question such as:

```text
Explain the cost-quality tradeoff of reasoning in Qwen3.5 4B and 9B for legal machine translation. Compare the different training and inference configurations and explain which configuration provides the best overall tradeoff.
```

can be submitted through the web interface.

The system retrieves relevant research-paper chunks, reranks them, generates a grounded answer, and returns the source papers used for the response.

---

## 🖥️ Frontend

The web interface is intentionally minimal.

It contains:

* **arXiv Research Assistant** title
* Research question input
* **Ask** button
* Loading state
* Error handling
* Generated answer
* Retrieved source papers

The frontend uses a single React component and plain CSS without additional UI libraries or state-management frameworks.

---

## 🖥️ Original Gradio Interface

The original Gradio interface is still available through:

```bash
python app.py
```

The FastAPI + React interface provides an alternative web-based interface while reusing the same underlying RAG pipeline.

---

## 🔮 Future Improvements

* 🔗 Make retrieved arXiv sources clickable
* 📄 Display paper titles and metadata alongside source IDs
* 📊 Expand the evaluation benchmark
* ⚡ Improve retrieval latency
* 💬 Add conversation history
* 🚀 Deploy the FastAPI backend and React frontend
* ☁️ Host the research assistant as a public application
* 📈 Add retrieval and generation performance monitoring

---

## 👨‍💻 Author

**Vansh Rotkar**

[GitHub](https://github.com/Vansh-glitch1505)
