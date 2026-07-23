import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

RETRIEVAL_K = 15     # over-retrieve on purpose, rerank narrows this down
FINAL_K = 4           # what actually goes into the final prompt

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------- Step 1: Load the persisted vector store ----------
def load_vectorstore():
    """Load the Chroma store that embed.py already built and saved to disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore


# ---------- Step 2: Retriever with query expansion ----------
def get_retriever(vectorstore):
    """Base similarity retriever wrapped with LangChain's MultiQueryRetriever,
    which generates multiple phrasings of the question and merges retrieved results."""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    return retriever


# ---------- Step 3: Hand-rolled reranker (your differentiator) ----------
RERANK_PROMPT = """You are ranking retrieved paper excerpts by relevance to a question.

Question: {question}

Excerpts:
{excerpts}

Return ONLY a JSON array of excerpt numbers, ordered from MOST to LEAST relevant.
Example: [3, 1, 4, 2]
"""

def rerank(query: str, docs: list) -> list:
    """LLM-based reranking — same approach as Day 5's rerank(), adapted
    to work on LangChain Document objects instead of plain strings."""
    if not docs:
        return []

    excerpts_text = "\n\n".join(
        f"[{i+1}] {doc.page_content[:500]}" for i, doc in enumerate(docs)
    )

    response = llm.invoke(RERANK_PROMPT.format(question=query, excerpts=excerpts_text))
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "", 1).strip()

    try:
        order = json.loads(raw)
        ranked_docs = [docs[i - 1] for i in order if 0 < i <= len(docs)]
    except (json.JSONDecodeError, IndexError):
        print("Warning: rerank parsing failed, falling back to original order")
        ranked_docs = docs

    return ranked_docs[:FINAL_K]


# ---------- Step 4: Full answer pipeline ----------
ANSWER_PROMPT = """Answer the question using ONLY the excerpts below. If the excerpts don't contain
enough information, say so clearly instead of guessing.

Cite which paper each fact comes from using the paper title in brackets, e.g. [Paper Title].

Question: {question}

Excerpts:
{context}

Answer:
"""

def answer_question(query: str):
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} chunks (pre-rerank)")

    reranked_docs = rerank(query, retrieved_docs)
    print(f"Reranked down to {len(reranked_docs)} chunks")

    context = "\n\n".join(
        f"[{doc.metadata.get('paper_title', 'Unknown')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    response = llm.invoke(ANSWER_PROMPT.format(question=query, context=context))
    return response.content


if __name__ == "__main__":
    query = "What is repetitive copying in long-context reasoning?"
    answer = answer_question(query)
    print("\n--- Answer ---")
    print(answer)