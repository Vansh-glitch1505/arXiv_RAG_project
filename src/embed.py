import os
import json
import time
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
    """Convert chunks into LangChain Documents, embed them in batches, store in Chroma."""
    docs = [
        Document(
            page_content=c["original_text"],
            metadata={
                "paper_id": c["paper_id"],
                "paper_title": c["paper_title"],
                "headline": c["headline"],
                "summary": c["summary"]
            }
        )
        for c in all_chunks
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    BATCH_SIZE = 50
    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            vectorstore.add_documents(batch)

        if i + BATCH_SIZE < len(docs):
            print("Waiting 60s to respect free-tier rate limit...")
            time.sleep(60)

    print(f"Stored {len(docs)} chunks in Chroma at {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    build_vectorstore(chunks)