"""
Nexa AI Chat Engine (V2.0)
------------------------------------------------------
Enhancements:
1. Strict Grounding: Forbids answering outside retrieved context.
2. Debug Tracing: Shows exactly which chunks were retrieved.
"""

import os
import sys
import nest_asyncio
import logging
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
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

# 1. Global Settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    trust_remote_code=True
)

# 2. LLM Configuration (Temperature 0 = Strict Factuality)
Settings.llm = Groq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0, 
    max_tokens=1024
)

def get_query_engine():
    """
    Builds a custom Query Engine with explicit retrieval steps.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[Error] PINECONE_API_KEY not found in .env")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(INDEX_NAME)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # --- STEP 1: RETRIEVAL (Wide Net) ---
    # Retrieve 20 chunks to ensure we catch spread-out tables/contexts
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )

    # --- STEP 2: RERANKING (Sniper Shot) ---
    # Re-score the 20 chunks and pick the top 5 most relevant ones
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL, 
        top_n=5
    )

    # --- STEP 3: SYSTEM PROMPT (Strict Grounding) ---
    system_prompt = (
        "You are the official Nexa Technical Assistant. "
        "Your goal is to answer questions strictly based on the provided retrieved context snippets.\n"
        "\n"
        "### STRICT RULES:\n"
        "1. **NO OUTSIDE KNOWLEDGE:** Do not use your internal training data. If the answer is not in the context, say 'I cannot find that information in the manual.'\n"
        "2. **CITE SOURCES:** When you provide a specific number (e.g., '29 PSI'), mention the section it came from (e.g., 'According to the Maintenance > Tires section...').\n"
        "3. **HANDLE TABLES:** The context contains Markdown tables. Read the rows carefully. If a user asks for 'Full Load', look for the 'Gross Weight' or 'Max Occupants' columns.\n"
        "4. **BE DIRECT:** Do not start with 'Based on the context provided'. Just state the answer.\n"
        "\n"
        "Context:\n"
    )

    # Assemble the Engine
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", # Concatenates chunks efficiently
        structured_answer_filtering=False
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer=response_synthesizer,
    )
    
    # Inject the system prompt override
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template_str": system_prompt + "{context_str}\n\nQuestion: {query_str}\nAnswer:"}
    )

    return query_engine

def main():
    print(f"--> Initializing Nexa AI ({LLM_MODEL})...")
    try:
        query_engine = get_query_engine()
        print("--> Ready! (Type 'exit' to quit)")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                print("Retrieving & Reasoning...", end="\r")
                
                # Execute Query
                response = query_engine.query(user_input)
                
                # --- DEBUG: SHOW SOURCES ---
                # This proves IF the chunking worked. If you see unrelated chunks here, 
                # then we adjust the retrieval. If you see the right chunks, the LLM is safe.
                print("\n[DEBUG: Top Retrieved Sources]")
                for node in response.source_nodes:
                    # Parse our custom metadata
                    meta = node.metadata
                    path = meta.get('context_path', 'Unknown Section')
                    score = f"{node.score:.3f}" if node.score else "N/A"
                    print(f" - [{score}] {path}: {node.text[:60]}...")
                print("-" * 50)

                # Streaming or Text Response
                print(f"AI: {response.response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[!] Error: {e}")

    except Exception as e:
        print(f"[!] Startup Error: {e}")

if __name__ == "__main__":
    main()
