import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone

# --- INIT ---
load_dotenv()

INDEX_NAME = "nexa-knowledge-base"
API_KEY = os.getenv("PINECONE_API_KEY")

if not API_KEY:
    print("[Error] PINECONE_API_KEY not found in .env")
    sys.exit(1)

def wipe_database():
    print(f"--> Connecting to Pinecone...")
    pc = Pinecone(api_key=API_KEY)
    
    # Check if index exists first
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"[!] Index '{INDEX_NAME}' does not exist. Nothing to wipe.")
        return

    index = pc.Index(INDEX_NAME)

    try:
        print(f"--> Wiping all vectors from '{INDEX_NAME}'...")
        print("    (This might take a few seconds...)")
        
        # delete_all=True removes every vector in the namespace
        # namespace='' targets the default namespace (where your data usually is)
        index.delete(delete_all=True, namespace='')
        
        print("--> SUCCESS: Database is now empty.")
        print(f"--> Index '{INDEX_NAME}' is still active and ready for new data.")
        
    except Exception as e:
        print(f"[!] Error during wipe: {e}")

if __name__ == "__main__":
    # Safety confirmation
    confirm = input(f"WARNING: This will delete ALL data in '{INDEX_NAME}'.\nAre you sure? (y/n): ")
    if confirm.lower() == 'y':
        wipe_database()
    else:
        print("Operation cancelled.")
