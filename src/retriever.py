from src.vector_store import search_similar, get_all_documents
from rank_bm25 import BM25Okapi

def retrieve(query: str, k=3):
    """
    Performs Hybrid Search using Pinecone (Dense) and BM25 (Sparse).
    Combines results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Vector Search
    vector_results = search_similar(query, k=k)
    
    # 2. BM25 Search
    docs = get_all_documents()
    bm25_results = []
    
    if docs:
        # Simple tokenization by whitespace
        tokenized_corpus = [doc.lower().split(" ") for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        
        for i, score in enumerate(doc_scores):
            bm25_results.append({
                "id": f"doc_{i}",
                "score": score,
                "text": docs[i]
            })
        
        # Sort and take top k
        bm25_results.sort(key=lambda x: x['score'], reverse=True)
        bm25_results = bm25_results[:k]
    
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    
    def add_to_rrf(results):
        for rank, item in enumerate(results):
            doc_id = item['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "text": item['text']}
            # RRF constant k=60 is standard
            rrf_scores[doc_id]["score"] += 1.0 / (60 + rank)
            
    add_to_rrf(vector_results)
    add_to_rrf(bm25_results)
    
    # Sort by combined RRF score
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)
    
    # Return top k texts
    return [doc['text'] for doc in sorted_docs[:k]]
