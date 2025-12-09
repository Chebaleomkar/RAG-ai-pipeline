from src.vector_store import search_similar

def retrieve(query: str, k=3):
    return search_similar(query, k)
