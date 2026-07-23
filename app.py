import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr
from answer import answer_question

def respond(message, history):
    """Gradio ChatInterface callback — takes the user message, returns the answer
    with source papers listed at the end."""
    answer, docs = answer_question(message)

    sources = list({doc.metadata.get("paper_title", "Unknown") for doc in docs})
    sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    return answer + sources_text


demo = gr.ChatInterface(
    respond,
    title="arXiv Research Assistant",
    description="Ask questions about recent NLP/AI research papers. Answers are grounded in retrieved paper excerpts, with sources cited below each response.",
    examples=[
        "What did the paper find about reasoning traces and translation quality?",
        "What happens when reasoning is enabled at inference but the model wasn't trained with reasoning?",
    ],
)

if __name__ == "__main__":
    demo.launch()