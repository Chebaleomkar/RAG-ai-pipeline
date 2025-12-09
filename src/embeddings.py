import google.generativeai as genai
from src.config import GEMINI_API_KEY

# Configure Gemini for embeddings
genai.configure(api_key=GEMINI_API_KEY)

def embed_text(text: str):
    """
    Generates embedding using Gemini's embedding model format.
    """
    model = "models/text-embedding-004"
    result = genai.embed_content(
        model=model,
        content=text
    )
    return result["embedding"]
