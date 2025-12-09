from src.retriever import retrieve
from src.llm_client import generate_answer

def ask_rag(query: str, k: int = 3):
    context_docs = retrieve(query, k)
    context_text = "\n".join(context_docs)

    prompt = f"""
You are a reliable and helpful assistant designed for Retrieval-Augmented Generation (RAG).

Your task:
- Answer the question using ONLY the information in the context.
- Do NOT use outside knowledge or assumptions.
- If the answer is not present in the context, reply kindly and clearly: "Not found."

Context:
{context_text}

Question:
{query}

Answer:
"""

    return generate_answer(prompt)
