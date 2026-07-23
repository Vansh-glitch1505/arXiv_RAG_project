import gradio as gr
from src.answer import answer_question, load_vectorstore, get_retriever, rerank

def respond(message, history):
    """Gradio ChatInterface callback — takes the user message, returns the answer
    with source papers listed at the end."""
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    retrieved_docs = retriever.invoke(message)
    reranked_docs = rerank(message, retrieved_docs)

    context = "\n\n".join(
        f"[{doc.metadata.get('paper_title', 'Unknown')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    from src.answer import llm, ANSWER_PROMPT
    response = llm.invoke(ANSWER_PROMPT.format(question=message, context=context))

    # Build a "Sources" footer from the reranked chunks' metadata
    sources = list({doc.metadata.get("paper_title", "Unknown") for doc in reranked_docs})
    sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    return response.content + sources_text


demo = gr.ChatInterface(
    respond,
    title="arXiv Research Assistant",
    description="Ask questions about recent NLP/AI research papers. Answers are grounded in retrieved paper excerpts, with sources cited below each response.",
    examples=[
        "What is repetitive copying in long-context reasoning?",
        "What methods reduce hallucination in RAG systems?",
    ],
)

if __name__ == "__main__":
    demo.launch()