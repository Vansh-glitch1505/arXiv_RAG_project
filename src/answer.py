import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

RETRIEVAL_K = 15
FINAL_K = 4

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------- Step 1: Load the persisted vector store ----------
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# ---------- Step 2: Hand-rolled query expansion (replaces MultiQueryRetriever) ----------
EXPANSION_PROMPT = """Generate 3 different versions of the question below, each phrased
differently, to help retrieve relevant documents from a vector database. Vary the wording
and phrasing, but keep the same underlying intent.

Original question: {question}

Return ONLY a JSON array of 3 strings, no markdown fences, no commentary.
Example: ["version 1", "version 2", "version 3"]
"""

def expand_query(query: str) -> list[str]:
    """Generate multiple phrasings of the question — hand-rolled replacement
    for LangChain's MultiQueryRetriever, since it's unavailable in this LangChain version."""
    response = llm.invoke(EXPANSION_PROMPT.format(question=query))
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "", 1).strip()

    try:
        variations = json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: query expansion parsing failed, using original query only")
        variations = []

    return [query] + variations   # always include the original query too


def multi_query_retrieve(vectorstore, query: str, k: int = RETRIEVAL_K):
    """Run retrieval across all query variations, then dedupe by content."""
    queries = expand_query(query)
    print(f"Expanded into {len(queries)} query variations")

    all_docs = []
    seen_content = set()

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_docs.append(doc)

    return all_docs


# ---------- Step 3: Hand-rolled reranker ----------
RERANK_PROMPT = """You are ranking retrieved paper excerpts by relevance to a question.

Question: {question}

Excerpts:
{excerpts}

Return ONLY a JSON array of excerpt numbers, ordered from MOST to LEAST relevant.
Example: [3, 1, 4, 2]
"""

def rerank(query: str, docs: list) -> list:
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

    retrieved_docs = multi_query_retrieve(vectorstore, query)
    print(f"Retrieved {len(retrieved_docs)} unique chunks (pre-rerank)")

    reranked_docs = rerank(query, retrieved_docs)
    print(f"Reranked down to {len(reranked_docs)} chunks")

    context = "\n\n".join(
        f"[{doc.metadata.get('paper_title', 'Unknown')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    response = llm.invoke(ANSWER_PROMPT.format(question=query, context=context))
    return response.content, reranked_docs


if __name__ == "__main__":
    query = "What happens when reasoning is enabled at inference but the model wasn't trained with reasoning?"
    answer, docs = answer_question(query)
    print("\n--- Answer ---")
    print(answer)