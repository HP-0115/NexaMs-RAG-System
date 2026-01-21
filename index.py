"""
Nexa Knowledge Base Indexer
---------------------------
This pipeline handles the "ETL" (Extract, Transform, Load) process for RAG.
1. LOADS markdown files from the 'data/parsed_docs' directory.
2. CHUNKS them using a Hybrid Strategy (Semantic + Recursive Fallback).
3. EMBEDS them using the BGE-Small model.
4. UPSERTS vectors to the Pinecone serverless database.

It includes safety checks to ensure no single chunk exceeds Pinecone's 40KB metadata limit.
"""

import os
import sys
import nest_asyncio
from typing import List
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- INIT ---
load_dotenv()
nest_asyncio.apply()

# --- CONFIGURATION ---
INDEX_NAME = "nexa-knowledge-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PARSED_DOCS_DIR = "data/parsed_docs"
# Pinecone has a strict 40KB limit for metadata. We set a safe buffer.
PINECONE_METADATA_LIMIT_BYTES = 30000 

# Global Settings
print(f"--> Loading Professional Embedding Model ({EMBEDDING_MODEL})...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    trust_remote_code=True
)

def get_pinecone_index():
    """
    Connects to Pinecone and ensures the index exists.
    If not, it creates a new Serverless index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[Error] PINECONE_API_KEY not found in .env")
        sys.exit(1)
    
    pc = Pinecone(api_key=api_key)
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"--> Creating new Index: {INDEX_NAME}...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384, # Matches BGE-Small dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as e:
            print(f"[!] Failed to create index: {e}")
            sys.exit(1)
            
    return pc.Index(INDEX_NAME)

def adaptive_chunking(documents: List[Document]) -> List[BaseNode]:
    """
    Hybrid Chunking Strategy:
    1. Attempt Semantic Chunking (Split by Meaning/Topic).
    2. Safety Check: If any chunk is too large for Pinecone, recursively split it
       using a standard SentenceSplitter.
    """
    print("--> 1. Running Semantic Chunking (Analyzing topic shifts)...")
    
    # Semantic Splitter uses embeddings to find "natural breaks"
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95, # Higher = stricter splitting
        embed_model=Settings.embed_model
    )
    
    initial_nodes = semantic_splitter.get_nodes_from_documents(documents)
    print(f"    Generated {len(initial_nodes)} semantic nodes.")

    print("--> 2. Running Adaptive Safety Check (Pinecone Compliance)...")
    final_nodes = []
    
    # Fallback splitter for massive nodes
    safety_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    for node in initial_nodes:
        # Check size of text + metadata roughly
        node_size = len(node.text.encode('utf-8'))
        
        if node_size > PINECONE_METADATA_LIMIT_BYTES:
            print(f"    [!] Found massive node ({node_size} bytes). Applying fallback splitting...")
            # Recursively split just this massive node
            sub_nodes = safety_splitter.get_nodes_from_documents([
                Document(text=node.text, metadata=node.metadata)
            ])
            final_nodes.extend(sub_nodes)
        else:
            final_nodes.append(node)
            
    print(f"    Final Node Count: {len(final_nodes)} (Safe for Pinecone)")
    return final_nodes

def load_and_index_data():
    """
    Main execution pipeline: Load MD -> Chunk -> Embed -> Upsert.
    """
    if not os.path.exists(PARSED_DOCS_DIR):
        print(f"[Error] Parsed docs directory '{PARSED_DOCS_DIR}' not found.")
        print("Did you run 'ingest.py' first?")
        return

    # 1. Setup Vector Store
    pinecone_index = get_pinecone_index()
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. Load Documents
    print("--> Reading markdown files...")
    all_documents = []
    files = [f for f in os.listdir(PARSED_DOCS_DIR) if f.endswith(".md")]
    
    if not files:
        print("[Warning] No .md files found to index.")
        return

    for filename in files:
        filepath = os.path.join(PARSED_DOCS_DIR, filename)
        # Convert "Grand_Vitara.md" -> "Grand Vitara"
        car_model = filename.replace(".md", "").replace("_", " ").title()
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        doc = Document(
            text=content,
            metadata={
                "car_model": car_model,
                "filename": filename,
                "type": "manual"
            }
        )
        all_documents.append(doc)
        print(f"    Loaded: {car_model}")

    # 3. Chunking
    nodes = adaptive_chunking(all_documents)

    # 4. Embedding & Upsert
    print(f"--> Embedding and Indexing {len(nodes)} chunks...")
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n--> SUCCESS! Knowledge Base is updated and live.")

if __name__ == "__main__":
    load_and_index_data()
