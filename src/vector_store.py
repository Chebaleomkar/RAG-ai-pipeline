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
    Supports list of strings or list of dicts with 'text' and 'metadata' keys.
    """
    vectors = []
    
    # Check if docs is strictly strings or dicts (checking first element)
    if not docs:
        print("No documents to add.")
        return

    processed_docs = [] # For local storage

    for i, doc in enumerate(docs):
        if isinstance(doc, dict) and "text" in doc:
            text_content = doc["text"]
            metadata = doc.get("metadata", {}).copy()
            
            # Sanitize metadata: Pinecone doesn't support None/null values
            # Remove keys with None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Ensure text is in metadata for retrieval
            metadata["text"] = text_content
        else:
            text_content = str(doc)
            metadata = {"text": text_content}
        
        # Store for local saving later
        processed_docs.append(text_content)

        emb = embed_text(text_content)
        vectors.append({
            "id": f"doc_{i}",
            "values": emb,
            "metadata": metadata
        })

    index.upsert(vectors)
    print("Documents added to Pinecone.")

    # Save documents locally for BM25
    import json
    import os
    
    DOCS_FILE = "documents.json"
    
    # Overwrite with current text content to ensure indices match Pinecone IDs (doc_i)
    # BM25 needs just the text list
    with open(DOCS_FILE, "w") as f:
        json.dump(processed_docs, f) # Save just the text list for now as per previous logic
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

    matches = []
    for match in result["matches"]:
        matches.append({
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"]["text"]
        })
    return matches
