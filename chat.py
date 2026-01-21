"""
Nexa AI Chat Engine
-------------------
This module handles the RAG (Retrieval-Augmented Generation) inference pipeline.
It connects to Pinecone, retrieves semantic chunks, reranks them for relevance,
and synthesizes answers using a Reasoning LLM.

Author: [Your Name]
Version: 1.0.0
"""

import os
import sys
import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.llms.groq import Groq
from pinecone import Pinecone

# --- INIT ---
load_dotenv()
nest_asyncio.apply()

# --- CONFIGURATION ---
INDEX_NAME = "nexa-knowledge-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 1. Configure Global Settings
# Embedding Model (Must match the one used in indexing)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    trust_remote_code=True
)

# LLM: Optimized for reasoning (GPT-OSS-20B via Groq)
Settings.llm = Groq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

def get_chat_engine():
    """
    Constructs the RAG engine with a two-stage retrieval process:
    1. Vector Search (Top 20)
    2. Cross-Encoder Reranking (Top 5)
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[Error] PINECONE_API_KEY not found in .env")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(INDEX_NAME)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # --- RERANKER ---
    # Filters false positives from the initial vector search
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL, 
        top_n=5
    )

    # --- SYSTEM PROMPT ---
    # Engineered for technical manual interpretation
    system_prompt = (
        "You are the Intelligent Manual Assistant for Nexa cars. "
        "Your job is to extract technical data from the provided context and answer the user's question directly."
        "\n\n"
        "### CRITICAL INFERENCE RULES:\n"
        "1. **Assume Intent:** If the user asks about 'Family' or 'Travel', they mean 'Gross Vehicle Weight' or 'Full Load'. Look for these terms in tables.\n"
        "2. **Read Tables:** Most answers are inside Markdown tables. If a row matches the user's query (e.g., '195/80 R15'), quote it.\n"
        "3. **Missing Header?** If you find a list of specs but no header, look at the chunk before it. Infer the context.\n"
        "\n"
        "### ANSWER FORMAT (Strict Order):\n"
        "1. **The Direct Answer:** Start with the specific number, step, or fact. (e.g., 'The recommended pressure is 26 PSI').\n"
        "2. **The Context:** Briefly mention which section this comes from (e.g., 'Based on the Full Load specifications...').\n"
        "3. **Safety Notes (Bottom):** Put any warnings at the very end as a footer.\n"
        "\n"
        "### TONE:\n"
        "- Confident and concise.\n"
        "- Do NOT say 'The manual does not state...' unless the page is blank."
    )
    
    return index.as_chat_engine(
        chat_mode="context",
        system_prompt=system_prompt,
        similarity_top_k=20,           # Wide Net: Catch huge tables
        node_postprocessors=[reranker] # Sniper Shot: Filter noise
    )

def main():
    print(f"--> Initializing Nexa AI ({LLM_MODEL})...")
    try:
        chat_engine = get_chat_engine()
        print("--> Ready! (Type 'exit' to quit)")
        print("-" * 50)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Streaming response for low-latency feel
                response = chat_engine.stream_chat(user_input)
                print("AI: ", end="", flush=True)
                for token in response.response_gen:
                    print(token, end="", flush=True)
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[!] Error during generation: {e}")
                print("-" * 50)

    except Exception as e:
        print(f"[!] Startup Error: {e}")

if __name__ == "__main__":
    main()
