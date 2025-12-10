from pinecone import Pinecone, ServerlessSpec
from src.embeddings import embed_text
from src.config import PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "rag-index"

# Create index only if it does NOT exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   # Required region for FREE TIER
        )
    )
    print(f"Created Pinecone index: {INDEX_NAME}")

# Connect to index
index = pc.Index(INDEX_NAME)


def add_documents(docs: list):
    """
    Adds documents to Pinecone with embeddings.
    """
    vectors = []
    for i, doc in enumerate(docs):
        emb = embed_text(doc)
        vectors.append({
            "id": f"doc_{i}",
            "values": emb,
            "metadata": {"text": doc}
        })

    index.upsert(vectors)
    print("Documents added to Pinecone.")

    # Save documents locally for BM25
    import json
    import os
    
    DOCS_FILE = "documents.json"
    
    # Overwrite with current docs to ensure indices match Pinecone IDs (doc_i)
    with open(DOCS_FILE, "w") as f:
        json.dump(all_docs, f)
    print(f"Documents saved locally to {DOCS_FILE} for BM25.")

def get_all_documents():
    """
    Retrieves all documents from the local store.
    """
    import json
    import os
    DOCS_FILE = "documents.json"
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    return []


def search_similar(query: str, k: int = 3):
    """
    Searches Pinecone for similar documents.
    """
    query_emb = embed_text(query)

    result = index.query(
        vector=query_emb,
        top_k=k,
        include_metadata=True
    )

    retrieved_texts = [match["metadata"]["text"] for match in result["matches"]]
    return retrieved_texts
