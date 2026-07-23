import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")


def load_chunks() -> list[dict]:
    """Load previously chunked data produced by ingest.py."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vectorstore(all_chunks: list[dict]):
    """Convert chunks into LangChain Documents, embed them, store in Chroma."""
    docs = [
        Document(
            page_content=c["original_text"],   # what actually gets embedded + retrieved
            metadata={
                "paper_id": c["paper_id"],
                "paper_title": c["paper_title"],
                "headline": c["headline"],
                "summary": c["summary"]
            }
        )
        for c in all_chunks
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Stored {len(docs)} chunks in Chroma at {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    build_vectorstore(chunks)