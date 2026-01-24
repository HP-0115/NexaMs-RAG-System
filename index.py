import os
import sys
import re
import logging
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

INDEX_NAME = "nexa-knowledge-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
PARSED_DOCS_DIR = "data/parsed_docs"

# Universal Safety Keywords (Do not split on these; keep them in the chunk)
SAFETY_KEYWORDS = {
    "WARNING", "CAUTION", "NOTICE", "NOTE", "TIP", "DANGER", "IMPORTANT"
}

# --- SETUP LLAMA INDEX & PINECONE ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    trust_remote_code=True
)

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY missing in .env")
        sys.exit(1)
    
    pc = Pinecone(api_key=api_key)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME, 
            dimension=EMBEDDING_DIM, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def is_file_in_database(index, filename):
    """Checks if a file has already been indexed to avoid duplicates."""
    try:
        # Query with a dummy vector just to check metadata filtering
        dummy_vector = [0.1] * EMBEDDING_DIM
        response = index.query(
            vector=dummy_vector, 
            filter={"filename": {"$eq": filename}}, 
            top_k=1, 
            include_metadata=False
        )
        return len(response['matches']) > 0
    except Exception:
        return False

# --- CORE LOGIC: HIERARCHICAL MARKDOWN CHUNKER ---
class HierarchicalBuilder:
    def __init__(self):
        # State tracking
        self.current_chapter = "General"
        self.current_topic = "Introduction"
        self.current_page = 1
        
        # Buffer for the current chunk being built
        self.chunk_buffer = []
        self.chunk_start_page = 1

    def is_chapter_header(self, text: str) -> bool:
        """Heuristic: Main Chapters are usually ALL CAPS (e.g., 'BEFORE DRIVING')."""
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        return clean_text.isupper() and len(clean_text) > 3

    def is_safety_header(self, text: str) -> bool:
        """Checks if header is a safety alert (WARNING/NOTE) that should remain in-text."""
        first_word = text.split()[0].upper().strip(":")
        return first_word in SAFETY_KEYWORDS

    def flush_chunk(self, car_model: str, filename: str) -> Optional[Dict]:
        """Packages the current buffer into a structured chunk dictionary."""
        text_content = "\n".join(self.chunk_buffer).strip()
        
        # Skip empty or noise chunks
        if len(text_content) < 50:
            return None

        # Create the Context Path string (The RAG Secret Sauce)
        context_path = f"{self.current_chapter} > {self.current_topic}"
        
        # Page Range Logic (e.g., "Page 5-6")
        if self.chunk_start_page == self.current_page:
            page_str = str(self.chunk_start_page)
        else:
            page_str = f"{self.chunk_start_page}-{self.current_page}"

        chunk_data = {
            "text": text_content,
            "metadata": {
                "car_model": car_model,
                "filename": filename,
                "chapter": self.current_chapter,
                "topic": self.current_topic,
                "page_number": page_str,
                "context_path": context_path
            }
        }
        return chunk_data

def process_file_hierarchical(filepath: str, filename: str, car_model: str) -> List[Document]:
    """
    Reads file line-by-line (M1 Memory Safe), builds a hierarchy tree, 
    and returns LlamaIndex Documents.
    """
    builder = HierarchicalBuilder()
    documents = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()

                # --- 1. NOISE REMOVAL (Just-in-Time) ---
                # Detect and ignore page markers to "heal" split sentences/tables
                page_match = re.match(r'^## Page (\d+)', stripped_line)
                if page_match:
                    builder.current_page = int(page_match.group(1))
                    continue  # SKIP: Do not add this line to the text buffer
                
                if stripped_line == '---':
                    continue # SKIP: Visual separators

                # --- 2. HEADER DETECTION ---
                if stripped_line.startswith('## '):
                    header_text = stripped_line[3:].strip()
                    
                    # A. Safety Header Glue: Downgrade to bold, keep in current chunk
                    if builder.is_safety_header(header_text):
                        builder.chunk_buffer.append(f"**{header_text}**")
                        continue

                    # B. Structural Split: Flush current buffer before starting new section
                    if builder.chunk_buffer:
                        chunk_data = builder.flush_chunk(car_model, filename)
                        if chunk_data:
                            # INJECT CONTEXT into the raw text for the embedding model
                            enriched_text = f"Context: {chunk_data['metadata']['context_path']}\n\n{chunk_data['text']}"
                            
                            doc = Document(
                                text=enriched_text,
                                metadata=chunk_data['metadata']
                            )
                            documents.append(doc)
                        
                        # Reset Buffer for the new section
                        builder.chunk_buffer = []
                        builder.chunk_start_page = builder.current_page

                    # C. Update Context Hierarchy
                    if builder.is_chapter_header(header_text):
                        builder.current_chapter = header_text
                        builder.current_topic = "General" # Reset topic on new chapter
                    else:
                        builder.current_topic = header_text # It's a sub-topic
                    
                    # Add the header itself to the text for readability
                    builder.chunk_buffer.append(f"# {header_text}")
                    continue

                # --- 3. ACCUMULATE CONTENT ---
                builder.chunk_buffer.append(stripped_line)

            # --- 4. FLUSH FINAL CHUNK ---
            if builder.chunk_buffer:
                chunk_data = builder.flush_chunk(car_model, filename)
                if chunk_data:
                    enriched_text = f"Context: {chunk_data['metadata']['context_path']}\n\n{chunk_data['text']}"
                    doc = Document(text=enriched_text, metadata=chunk_data['metadata'])
                    documents.append(doc)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

    return documents

# --- MAIN EXECUTION ---
def load_and_index_data():
    if not os.path.exists(PARSED_DOCS_DIR):
        print(f"Error: Directory '{PARSED_DOCS_DIR}' not found.")
        return

    pinecone_index = get_pinecone_index()
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    files = [f for f in os.listdir(PARSED_DOCS_DIR) if f.endswith(".md")]
    print(f"--> Found {len(files)} manuals.")

    for filename in files:
        car_model = filename.replace(".md", "").replace("_", " ").title()
        filepath = os.path.join(PARSED_DOCS_DIR, filename)

        if is_file_in_database(pinecone_index, filename):
            print(f" [SKIP] {car_model} already in DB.")
            continue

        print(f" [Indexing] {car_model} via Hierarchical Strategy...")
        
        # PROCESS
        documents = process_file_hierarchical(filepath, filename, car_model)
        
        if documents:
            print(f"    -> Created {len(documents)} semantic chunks.")
            
            # INDEX
            VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                show_progress=False
            )
            print(f"    -> {car_model} Indexing Complete.")
        else:
            print("    -> No valid chunks found.")

    print("\n--> Sync Complete.")

if __name__ == "__main__":
    load_and_index_data()
