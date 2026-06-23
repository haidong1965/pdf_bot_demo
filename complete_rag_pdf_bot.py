# -*- coding: utf-8 -*-

import sys
import os
import time 
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError 
from chromadb import Client, Collection 
from sentence_transformers import SentenceTransformer 

try:
    from langchain_community.document_loaders import PyPDFLoader 
except ImportError as e:
    print(f"?? CRITICAL LIBRARY ERROR: Failed to load PDF loader. Did you reinstall dependencies? Error details: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter 
except ImportError:
    print("?? WARNING: Could not find 'langchain_text_splitters'. Attempting fallback...")
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter # Fallback attempt 1
    except ImportError as e:
         print(f"?? CRITICAL FAILURE: Cannot locate Text Splitter class in any known path. Error: {e}")

# --- CONFIGURATION MANAGEMENT (Centralized) ---

LOCAL_LM_STUDIO_BASE_URL = "http://192.168.1.81:1234/v1" 
PDF_DIR_NAME = 'pdf' # Use a constant name for the directory

# --- INITIALIZATION & GLOBAL SETUP ---

current_directory = Path.cwd()
full_path = current_directory / PDF_DIR_NAME
PDF_DIR = full_path.resolve() 

print("==============================================")
print("--- Initializing RAG Components ---")
print(f"Checking for PDFs in: {PDF_DIR}")
print("==============================================\n")


local_model = None
try:
    EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5' 
    local_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"? Local Embedding Model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"? ERROR initializing local model: {e}. Embedding functionality will be disabled.")

# Initialize ChromaDB Client (PersistentClient is generally preferred for stateful apps)
chroma_client = None
try:
    # Use a specific, consistent directory for the database path
    CHROMA_PATH = "./chroma_db" 
    chroma_client = Client() # Using default client for simplicity unless persistence is key
    print(f"? ChromaDB Client initialized successfully.")
except Exception as e:
    print(f"? ERROR initializing ChromaDB client: {e}")

def generate_embeddings(chunks: list, model: SentenceTransformer):
    """Generates embeddings for a list of chunks using the provided local BGE model."""
    if not model:
        print("\n?? Cannot generate embeddings: Local Embedding Model is unavailable.")
        return None

    texts_to_embed = [chunk.page_content for chunk in chunks]
    print(f"\n--- Generating {len(texts_to_embed)} Embeddings using BGE Model ---")
    
    try:
        # The encode method handles the batch encoding efficiently
        embeddings = model.encode(texts_to_embed)
        print("? Embedding generation complete.")
        return embeddings

    except Exception as e:
        print(f"? EMBEDDING FAILED (BGE): An error occurred during encoding. Details: {e}")
        return None


def upsert_to_vectorstore(chunks, embeddings, metadatas, client_instance: Client):
    """Upserts document content and embeddings into ChromaDB."""
    if not chunks or len(embeddings) == 0:
        print("?? Cannot upsert embeddings: No data available.")
        return False

    COLLECTION_NAME = "rag_knowledge_base" 
    try:
        # Use the passed client instance to ensure consistency
        collection = client_instance.get_or_create_collection(name=COLLECTION_NAME)

        # 1. Generate GUARANTEED Unique IDs for ALL chunks (Crucial step)
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # 2. Prepare Document Content List
        document_contents = [getattr(chunk, 'page_content', str(chunk)) for chunk in chunks]

        # 3. Perform the Upsert
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(), 
            metadatas=metadatas, 
            documents=document_contents
        )

        print("? SUCCESS: All data written to ChromaDB with unique IDs.")
        return True

    except Exception as e:
        print(f"? CRITICAL FAILURE: A database write error occurred during upsert. Details: {e}")
        return False


def generate_answer(user_query: str, context: list[dict]) -> str:
    """
    Uses an LLM to synthesize a final answer based on the retrieved context and the original query.
    """
    if not context:
        return "I could not find any relevant information in the knowledge base to answer your question."

    # --- 1. Format the Context for the Prompt (KEEP THIS) ---
    context_text = "\n---\n".join(
        [f"Source Document Snippet:\n{item['document']}\n\nMetadata: {item['metadatas']}" 
         for item in context]
    )

    # --- 2. Define the System Prompt (KEEP THIS) ---
    system_prompt = (
        "You are an expert Research Assistant. Your task is to answer the user's "
        "question based ONLY on the provided CONTEXT. "
        "If the context does not contain enough information to answer the question, "
        "you MUST respond with: 'I do not have enough information in the provided documents to answer that.' "
        "Do not use any external knowledge."
    )
    openai_client =  OpenAI(
        api_key="lm-studio",
        base_url=LOCAL_LM_STUDIO_BASE_URL 
        )

    messages = [
        {"role": "system", "content": system_prompt},  # Sets the rules first
        {
            "role": "user", 
            "content": f"""
            CONTEXT DOCUMENT:\n---\n{context}\n---\n\n
            Based ONLY on the context provided above, please answer the following question: '{user_query}'
            """
        }
    ]

    print("\n[--- Calling LM Studio API ---]")

    try:
        # The client sends the request to localhost instead of OpenAI's servers
        response = openai_client.chat.completions.create(
            model="llama-3.2-3b-instruct", # You can use the model name you are running
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
            error_msg = str(e)
            raise e


class Indexer:
    """Handles loading, chunking, embedding, and storing documents."""
    def __init__(self, pdf_dir: Path, model: SentenceTransformer, chromadb_client: Client):
        self.pdf_dir = pdf_dir
        self.model = model
        self.chroma_client = chromadb_client
        self.collection = None

    def _load_documents(self) -> list[any]:
        """Stage 1: Load all PDFs into LangChain Document objects."""
        all_documents = []
        pdf_paths = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_paths:
            print("[FATAL] NO FILES FOUND: Please check the directory.")
            return []

        print(f"\n[STAGE 1/4] Found {len(pdf_paths)} PDF files to process.")
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(str(pdf_path)) 
                docs = loader.load()
                all_documents.extend(docs)
                print(f"   -> Successfully loaded {len(docs)} pages from {pdf_path.name}")
            except Exception as e:
                 print(f"?? SKIPPED {pdf_path.name}: Failed to load PDF. Error: {e}")
        return all_documents

    def index_data(self):
        """Runs the full ingestion pipeline."""
        
        # Stage 2: Chunking
        documents = self._load_documents()
        if not documents:
            print("Indexing halted due to no documents loaded.")
            return False

        print("\n[STAGE 2/4] Chunking Documents...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) 
            chunks = text_splitter.split_documents(documents)
            print(f"? SUCCESS: Created {len(chunks)} manageable chunks.")
        except Exception as e:
            print(f"? CHUNKING FAILED: Cannot proceed. Error: {e}")
            return False

        # Stage 3: Embedding & Upserting
        if not self.model:
            print("\n[?? HALT] Indexing failed because the local embedding model is unavailable.")
            return False

        print("\n[STAGE 3/4] Generating Embeddings...")
        embeddings = generate_embeddings(chunks, self.model)
        
        if embeddings is None or embeddings.size == 0:
            print("[?? HALT] Embedding generation failed. Cannot proceed to storage.")
            return False

        print("\n[STAGE 4/4] Storing Data...")
        # Extract metadata list efficiently using a list comprehension
        metadata_list = [c.metadata for c in chunks] 
        
        success = upsert_to_vectorstore(chunks, embeddings, metadata_list, self.chroma_client)
            
        if success:
            print("\n===================================================")
            print("? INGESTION PIPELINE COMPLETE! Data is indexed.")
            print("===================================================")
        else:
            print("\n===================================================")
            print("?? INGESTION FAILED during database write.")
            print("===================================================")
        return success


class RAGSystem:
    """Manages the entire retrieval and generation process."""
    def __init__(self, query_user: str):
        self.query_user = query_user
        # Use the global/pre-initialized client instance
        self.client = chroma_client 
        self.collection: Collection | None = None

    def connect_and_retrieve(self) -> list[dict] | None:
        """Phase 1, 2, & 3 combined: Connects, embeds the query, and retrieves context."""
        if not self.client:
            print("Error: Chroma client is not initialized.")
            return None

        try:
            # Ensure collection exists before querying (Critical step for Phase 1)
            COLLECTION_NAME = "rag_knowledge_base" 
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"[Phase 1/3] Database connection established to '{COLLECTION_NAME}'.")

            # Phase 2: Embedding Generation for the Query (Must use the same model!)
            if not local_model:
                raise RuntimeError("Local embedding model is not available for query.")

            embeddings_result = local_model.encode(self.query_user)
            query_list = [embeddings_result] 
            print("[Phase 2/3] Query Embedding generated successfully.")

            # Phase 3: Retrieval
            print("\n[Phase 3/3] Querying Knowledge Base...")
            results = self.collection.query(
                query_embeddings=query_list,  
                n_results=5,                       
                include=['documents', 'metadatas'] 
            )

            if not results or ('documents' not in results):
                 print("Warning: Retrieval returned no results.")
                 return None
            
            # Context Extraction (Cleaned up mapping)
            retrieved_context = []
            for i, doc_text in enumerate(results['documents']):
                meta = results['metadatas'][i]
                retrieved_context.append({'document': doc_text, 'metadatas': meta})

            print("? Phase 3 Success: Context retrieved successfully.")
            return retrieved_context

        except Exception as e:
            print(f"? CRITICAL FAILURE in Retrieval Process: {type(e).__name__}: {e}")
            return None


# =====================================================================
#                       MAIN EXECUTION BLOCK                    #
# =====================================================================

if __name__ == "__main__":
    # --- 1. RUN INGESTION PIPELINE (Indexing) ---
    print("\n\n" + "#"*10 + " STARTING INDEXING PHASE " + "#"*10)
    indexer = Indexer(PDF_DIR, local_model, chroma_client)
    indexing_successful = indexer.index_data()

    # --- 2. RUN QUERY PIPELINE (Retrieval & Generation) ---
    print("\n\n" + "#"*10 + " STARTING QUERY PHASE " + "#"*10)

    if indexing_successful:
        rag_system = RAGSystem(query_user="What is the formula used to calculate the OH processing stage index stgIx?")
        context = rag_system.connect_and_retrieve()
        
        final_answer = "Could not generate an answer due to a critical system failure."

        if context:
            try:
                # Pass the query and the retrieved context object directly
                final_answer = generate_answer(user_query=rag_system.query_user, context=context)
            except Exception as e:
                 print(f"\n? CRITICAL FAILURE during Answer Generation: {e}")

        print("\n" + "="*60)
        print("? RAG PIPELINE EXECUTION FINISHED ?")
        print(f"FINAL QUERY: {rag_system.query_user}")
        print("-" * 20)
        print(final_answer)
        print("="*60)

    else:
        print("\n\n*** Cannot run query phase because the data indexing failed. ***")
