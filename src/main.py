import os
from src.vector_store import add_documents
from src.rag_engine import ask_rag
from src.pdf_processor import PDFProcessor

# Validated PDF path
PDF_PATH = os.path.join(os.getcwd(), "attention-is-all-you-need-Paper.pdf")

if os.path.exists(PDF_PATH):
    print(f"Processing {PDF_PATH}...")
    processor = PDFProcessor(PDF_PATH)
    
    # 1. Extract text
    raw_text = processor.extract_text()
    
    # 2. Clean text
    cleaned_text = processor.clean_text(raw_text)
    
    # 3. Chunk text
    # Using larger chunk size for paper context
    chunks = processor.chunk_text(cleaned_text, chunk_size=1000, overlap=200)
    
    # 4. Add to Vector Store
    # chunks are list of dicts: {'text': ..., 'metadata': ...}
    add_documents(chunks)
    
    question = input("Please enter your question: ")
    answer = ask_rag(question)
    print(f"Answer: {answer}")

else:
    print(f"Error: PDF file not found at {PDF_PATH}")
    # Fallback/Exit
    exit(1)
