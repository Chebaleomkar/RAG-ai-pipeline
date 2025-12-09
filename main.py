import google.generativeai as genai 
from groq import Groq
import chromadb
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMIN_API_KEY"))

def embed_text(texts):
    model = "models/text-embedding-004"
    embeddings = genai.embed_content(
        model=model,
        content=texts
    )["embedding"]
    return embeddings

chroma_client = chromadb.PersistentClient(path="chroma_data")
collection = chroma_client.get_or_create_collection(name="docs")

docs = [
    "To reset your router, hold the reset button for 10–15 seconds.",
    "Wi-Fi light blinking indicates the router is trying to connect.",
    "Factory reset erases all custom passwords and settings."
]

existing_ids = set(collection.get()['ids'])

for i,doc in enumerate(docs):
    doc_id = f"doc_{i}"
    if doc_id not in existing_ids:
        emb = embed_text(doc)
        collection.add(
            documents=[doc],
            embeddings=[emb],
            metadatas=[{"source": f"doc_{i}"}],
            ids=[f"doc_{i}"]
        )

def retrieve(query,k=3):
    query_emb = embed_text(query)
    results= collection.query(
        query_embeddings=[query_emb],
        n_results=k
    )
    retrieved_docs = results["documents"][0]
    return retrieved_docs



client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_rag(query):
    context_docs = retrieve(query, k=3)
    context_text = "\n".join(context_docs)

    prompt = f"""
You are a reliable and helpful assistant designed for Retrieval-Augmented Generation (RAG).

Your task:
- Answer the question **using only the information provided in the context**.
- **Do NOT use outside knowledge**, assumptions, or guesses.
- If the answer is not present in the context, reply kindly and clearly: "Not found."

Guidelines:
- Stay factual and concise.
- Cite or reference the specific context sentences naturally (no numbers needed).
- If multiple context items are relevant, merge them into one clear answer.
- If context contains irrelevant information, ignore it.

Context:
{context_text}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


print(ask_rag("How do I reset my wifi router?"))
