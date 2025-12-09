import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables immediately when this file is imported

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")