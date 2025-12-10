import PyPDF2
from typing import List, Dict
import re

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        """Extract all text from PDF"""
        text = ""
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"📄 Extracting from {len(pdf_reader.pages)} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                # Add page number as metadata marker
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                   overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Extract page number from text
            page_match = re.search(r'--- Page (\d+) ---', chunk_text)
            page_num = int(page_match.group(1)) if page_match else None
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'chunk_id': i // (chunk_size - overlap),
                    'page': page_num,
                    'source': self.pdf_path
                }
            })
        
        print(f"✂️ Created {len(chunks)} chunks")
        return chunks
