from src.vector_store import add_documents
from src.rag_engine import ask_rag

docs = [
    "To reset your router, hold the reset button for 10–15 seconds.",
    "Wi-Fi light blinking indicates the router is trying to connect.",
    "Factory reset erases all custom passwords and settings."
]

add_documents(docs)

print(ask_rag("How do I reset my wifi router?"))
