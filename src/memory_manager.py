import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import dotenv
dotenv.load_dotenv()

class MemoryManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Using a lightweight local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_memory"
        )

    def store_context(self, text, metadata=None):
        """Stores a piece of research context or conversation."""
        if metadata is None:
            metadata = {}
        
        metadata["timestamp"] = datetime.now().isoformat()
        
        self.vector_db.add_texts(
            texts=[text],
            metadatas=[metadata]
        )
        # Chroma handles persistence automatically in newer versions, 
        # but check for manual persist if using older community versions.

    def fetch_relevant_history(self, query, k=5):
        """Searches for the most relevant past research context."""
        try:
            results = self.vector_db.similarity_search(query, k=k)
            print(f"DEBUG: Memory search for '{query}' returned {len(results)} results.")
            return results
        except Exception as e:
            print(f"ERROR: Memory search failed: {e}")
            return []

if __name__ == "__main__":
    # Quick test
    memory = MemoryManager()
    memory.store_context("I am researching the effects of gravitational waves on star formation.", {"topic": "astrophysics"})
    print("Stored context.")
    
    hits = memory.fetch_relevant_history("Tell me about stars and gravity")
    for hit in hits:
        print(f"Found: {hit.page_content} (Meta: {hit.metadata})")
