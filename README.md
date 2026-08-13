# 📚 arXiv RAG Research Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to ask questions about academic research papers from **arXiv**.

The project combines **semantic chunking, Gemini embeddings, ChromaDB, query expansion, vector retrieval, LLM reranking, and grounded answer generation** to provide relevant answers with source papers.

## ✨ Features

* 📄 Extracts and processes research papers from PDFs
* 🧩 **LLM-based semantic chunking** to preserve meaningful sections
* 🧠 **Gemini embeddings** for semantic search
* 🗃️ **ChromaDB** for persistent vector storage
* 🔀 **Query expansion** to improve retrieval coverage
* 🔍 Multi-query vector retrieval
* 🎯 **LLM-based reranking** of retrieved chunks
* 📚 Displays source papers used for the answer
* 📊 Evaluation using **Retrieval Accuracy, MRR, NDCG@4, and Fact Recall**
* 🖥️ Gradio-based chat interface

## 🛠️ Tech Stack

**Language:** Python

**LLM:** Google Gemini 2.5 Flash

**Embeddings:** Gemini Embedding

**RAG / Vector Store:** LangChain, ChromaDB

**Document Processing:** PyPDF

**Interface:** Gradio

**Data / Evaluation:** JSON, JSONL

## 📁 Project Structure

```text
arXiv_RAG_project/
│
├── app.py              # Gradio application
├── src/
│   ├── ingest.py       # PDF processing & semantic chunking
│   ├── embed.py        # Embedding generation & ChromaDB
│   ├── answer.py       # Retrieval, reranking & answer generation
│   ├── eval.py         # RAG evaluation
│   └── check_models.py # Gemini model utility
│
├── data/
│   ├── raw/            # Research papers
│   ├── chunks.json     # Processed chunks
│   └── chroma_db/      # Vector database
│
└── evals/
    └── eval_set.jsonl  # Evaluation dataset
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vansh-glitch1505/arXiv_RAG_project.git
cd arXiv_RAG_project
```

### 2. Install dependencies

```bash
pip install arxiv requests pypdf python-dotenv langchain-chroma langchain-google-genai langchain-core gradio google-genai
```

### 3. Add your API key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

### 4. Run the pipeline

Process the papers:

```bash
python src/ingest.py
```

Generate embeddings and build the vector database:

```bash
python src/embed.py
```

Start the research assistant:

```bash
python app.py
```

### 5. Run evaluation

```bash
python src/eval.py
```

## 📊 Evaluation

The project includes a 15-question evaluation set and measures:

* **Retrieval Accuracy** – whether the relevant paper is retrieved
* **MRR** – how highly the relevant paper is ranked
* **NDCG@4** – quality of the top retrieved results
* **Fact Recall** – whether expected facts appear in the generated answer

## 🔮 Future Improvements
* Better User Interface
* Larger evaluation benchmark
* Deployment as a public research assistant

## 👨‍💻 Author

**Vansh Rotkar**

[GitHub](https://github.com/Vansh-glitch1505)
