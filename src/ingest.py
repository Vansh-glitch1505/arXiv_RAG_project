import arxiv
import requests
import os
import time
import json
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------- Step 1: Fetch papers from arXiv ----------
def download_pdf(pdf_url: str, filepath: str):
    """Download a PDF directly via requests — bypasses arxiv package's
    own download method, since that's version-inconsistent."""
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(response.content)


def fetch_papers(max_results: int = 10):
    search = arxiv.Search(
        query="cat:cs.CL",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    client = arxiv.Client()

    papers = []
    for result in client.results(search):
        print(result.title)
        paper_id = result.get_short_id()
        filepath = os.path.join(RAW_DIR, f"{paper_id}.pdf")

        download_pdf(result.pdf_url, filepath)
        print(f"Saved to {filepath}")
        print("---")

        papers.append({
            "id": paper_id,
            "title": result.title,
            "pdf_path": filepath
        })

    return papers


# ---------- Step 2: Extract text from PDF ----------
def extract_text(pdf_path: str) -> str:
    """Pull raw text out of a downloaded PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ---------- Step 3: Chunk the paper (LLM-based semantic chunking) ----------
def split_into_sections(text: str, max_words: int = 2500, overlap_words: int = 250) -> list[str]:
    """Coarse length-based pre-split with overlap, so ideas straddling
    a section boundary aren't cut in half before the LLM even sees them."""
    words = text.split()
    sections = []
    step = max_words - overlap_words

    for i in range(0, len(words), step):
        section = " ".join(words[i:i + max_words])
        sections.append(section)
        if i + max_words >= len(words):
            break

    return sections


CHUNK_PROMPT = """You are splitting a section of an academic paper into semantic chunks for a retrieval system.

Rules:
- Split at natural idea/paragraph boundaries, not fixed character counts
- Do NOT split a table row, equation, or single claim across two chunks
- Each chunk should be understandable on its own, without needing the previous chunk for context
- For each chunk, provide: a short headline, a 1-sentence summary, and the original text verbatim

Return ONLY a JSON array, no markdown fences, no commentary. Format:
[{{"headline": "...", "summary": "...", "original_text": "..."}}]

Section text:
{section_text}
"""


def chunk_section(section_text: str) -> list[dict]:
    """Run LLM-based chunking on ONE section (not the whole paper)."""
    response = llm.invoke(CHUNK_PROMPT.format(section_text=section_text))
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw.replace("json", "", 1).strip()

    try:
        chunks = json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: failed to parse chunk JSON, skipping this section")
        return []

    return chunks


def chunk_paper(paper_text: str, paper_id: str, paper_title: str) -> list[dict]:
    """Full chunking pipeline for one paper: coarse split → LLM chunk each section → tag with metadata."""
    sections = split_into_sections(paper_text)
    all_chunks = []

    for i, section in enumerate(sections):
        print(f"  Chunking section {i+1}/{len(sections)}...")
        chunks = chunk_section(section)
        for c in chunks:
            c["paper_id"] = paper_id
            c["paper_title"] = paper_title
        all_chunks.extend(chunks)
        time.sleep(2)

    return all_chunks


def load_existing_papers() -> list[dict]:
    """Rebuild the papers list from PDFs already downloaded in data/raw/,
    so you don't re-hit the arXiv API while testing chunking."""
    papers = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            paper_id = filename.replace(".pdf", "")
            papers.append({
                "id": paper_id,
                "title": paper_id,   # title isn't stored on disk, fallback to ID
                "pdf_path": os.path.join(RAW_DIR, filename)
            })
    return papers


# ---------- Run ----------
if __name__ == "__main__":
    papers = load_existing_papers()
    papers = papers[:3]
    print(f"\nLoaded {len(papers)} existing papers\n")

    all_chunks = []
    chunks_path = os.path.join(BASE_DIR, "data", "chunks.json")

    for paper in papers:
        print(f"Processing: {paper['title']}")
        text = extract_text(paper["pdf_path"])
        chunks = chunk_paper(text, paper["id"], paper["title"])
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks\n")

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)   # save after every paper

    print(f"\nTotal chunks across all papers: {len(all_chunks)}")
    print(f"Saved chunks to {chunks_path}")