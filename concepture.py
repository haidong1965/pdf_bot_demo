# -*- coding: utf-8 -*-
"""
Comprehensive RAG Pipeline Script (Version 5 - Finalized Dependency Structure)
=============================================================================
?? IMPORTANT: Ensure you ran the dependency reinstall steps above before running this!
=============================================================================
"""
from sentence_transformers import SentenceTransformer 
import os
from pathlib import Path
import time 
import chromadb # Make sure to import the library

# =====================================================================
# ?? ALL NECESSARY IMPORTS (Using robust fallback imports)
# =====================================================================
try:
    from langchain_community.document_loaders import PyPDFLoader # Loader for PDFs
except ImportError as e:
    print(f"?? CRITICAL LIBRARY ERROR: Failed to load PDF loader. Did you reinstall dependencies? Error details: {e}")

try:
    # *** CRITICAL FIX ATTEMPT ***: Trying the most common fallback path for TextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter 
except ImportError:
    print("?? WARNING: Could not find 'langchain_text_splitters'. Falling back to explicit core import.")
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter # Fallback attempt 1
    except ImportError as e:
         print(f"?? CRITICAL FAILURE: Cannot locate Text Splitter class in any known path. Error: {e}")


# API Clients (These rely on the 'openai' library installed via pip)
from openai import OpenAI 

# Vector Store Client
from chromadb import Client 


# =====================================================================
# ?? EXTERNAL CLIENT INITIALIZATION
# =====================================================================

# =====================================================================
# ?? EXTERNAL CLIENT INITIALIZATION (Run this block after imports)
# =====================================================================

# --- LOCAL EMBEDDING MODEL INITIALIZATION ---
try:
    # Using the recommended BGE model for better performance than MiniLM
    EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5' 
    local_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"? Local Embedding Model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"? ERROR initializing local model: {e}. Embedding functionality will fail.")
    local_model = None

# REMOVE OR COMMENT OUT THE LINE: openai_client = OpenAI() 



vector_store_client = None # This assignment is now safe because it's outside any open try/catch block!
try:
    # Initialize ChromaDB client for local use
    vector_store_client = Client() 
    print("? ChromaDB Client initialized successfully (local instance).")
except Exception as e:
    print(f"? ERROR initializing ChromaDB client: {e}")


# =====================================================================
# !!! CONFIGURATION AREA (MANUALLY EDIT THESE) !!!
# =====================================================================

PDF_DIR = Path(r"C:\ai\pdf") # <<< ?? CHANGE THIS PATH! ??

EMBEDDING_MODEL_NAME = "text-embedding-ada-002" 
# =====================================================================
# !!! CORE FUNCTIONS (REMAINING LOGIC - UNCHANGED) !!!
# =====================================================================

def generate_embeddings(chunks):
    """Generates embeddings for a list of chunks using the local BGE model."""
    if not local_model:
        print("\n?? Cannot generate embeddings: Local Embedding Model is unavailable.")
        return None

    texts_to_embed = [chunk.page_content for chunk in chunks]
    
    # *** FIX APPLIED HERE ***
    # Use the constant name instead of calling a non-existent method on the object
    print(f"\n--- Generating {len(texts_to_embed)} Embeddings using BGE Model ({EMBEDDING_MODEL_NAME}) ---")
    
    try:
        embeddings = local_model.encode(texts_to_embed)
        print("? Embedding generation complete.")
        return embeddings

    except Exception as e:
        print(f"? EMBEDDING FAILED (BGE): An error occurred during encoding. Details: {e}")
        return None


def upsert_to_vectorstore(chunks, embeddings, metadatas):
    """Upserts document content and embeddings into ChromaDB using guaranteed unique IDs."""
    if not chunks or len(embeddings) == 0:
        print("?? Cannot upsert embeddings: No data available.")
        return False

    try:
        COLLECTION_NAME = "rag_knowledge_base" 
        client = chromadb.Client() # Adjust client initialization if needed
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        # --- 1. Generate GUARANTEED Unique IDs for ALL chunks (The Final Fix!) ---
        ids = []
        for i, chunk in enumerate(chunks):
            # *** FINAL FIX: Relying solely on the loop index 'i' ensures no collisions within this run. ***
            unique_id = f"chunk_{i}" 
            ids.append(unique_id)

        # --- 2. Prepare Document Content List ---
        document_contents = []
        for chunk in chunks:
            # Robust content access remains the same
            content = getattr(chunk, 'page_content', str(chunk)) 
            document_contents.append(content)

        # --- 3. Perform the Upsert with IDs, Documents, Embeddings, and Metadata ---
        collection.add(
            ids=ids, # <--- Passing the guaranteed unique list of IDs!
            embeddings=embeddings.tolist(), 
            metadatas=metadatas, 
            documents=document_contents
        )

        print("? SUCCESS: All data written to ChromaDB with unique IDs.")
        return True

    except Exception as e:
        # Catching the final error!
        print(f"? CRITICAL FAILURE: A database write error occurred. Details: {e}")
        return False

# =====================================================================
#                       MAIN EXECUTION FUNCTION                    #
# =====================================================================

def run_pipeline():
    """Orchestrates the entire robust document ingestion process."""
    global documents, chunks
    documents = [] 
    chunks = []    

    print("\n" + "=" * 80)
    print("?? STARTING RAG PIPELINE: Document Ingestion Pipeline")
    print("=" * 80)

    # --- STEP 1 & 2: LOAD DOCUMENTS (PDFs) ---
    all_pdf_paths = list(PDF_DIR.glob("*.pdf"))

    if all_pdf_paths:
        print(f"\n[STAGE 1/4] Found {len(all_pdf_paths)} PDF files to process.")
        for pdf_path in all_pdf_paths:
            try:
                loader = PyPDFLoader(str(pdf_path)) 
                docs = loader.load()
                documents.extend(docs)
                print(f"   -> Successfully loaded {len(docs)} pages from {pdf_path.name}")
            except Exception as e:
                 print(f"?? SKIPPED {pdf_path.name}: Failed to load PDF. Error: {e}")
    else:
        print("\n[FATAL] NO FILES FOUND: Please check the path defined in PDF_DIR.")

    # --- STEP 3: CHUNKING ---
    if documents and len(documents) > 0:
        print("\n[STAGE 2/4] Chunking Documents...")
        try:
            # Use the class that was successfully imported above
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) 
            chunks = text_splitter.split_documents(documents)
            print(f"? SUCCESS: Created {len(chunks)} manageable chunks.")
        except Exception as e:
            print(f"? CHUNKING FAILED: Cannot proceed. Error: {e}")
            chunks = [] 
    else:
        print("\n[SKIP] No documents were loaded; skipping chunking.")

    # --- STEP 4: EMBEDDING & UPSERTING (The final, critical step) ---
        # ... inside run_pipeline() ...
    if chunks and local_model: # Check if the model object exists
        print("\n[STAGE 3/4] Generating Embeddings...")
        embeddings = generate_embeddings(chunks) # <<< NOTICE: No client argument passed!
            
    if embeddings is None or embeddings.size == 0:
        print("\n[?? HALT] Embedding generation failed. Cannot proceed to storage.")
        return # Exit the pipeline gracefully
    else:
        # ... rest of the function remains largely the same, but remove dependency on OpenAI initialization checks
        print("[STAGE 4/4] Storing Data...")
        # 1. Extract the list of metadata dictionaries from your chunks list
        metadata_list = [c.metadata for c in chunks] 
        success = upsert_to_vectorstore(chunks, embeddings, metadata_list) 
            
        if success:
            print("\n===================================================")
            print("? PIPELINE COMPLETE! All data successfully ingested!")
            print("===================================================")
        else:
            print("\n===================================================")
            print("?? PIPELINE HALTED: Check initialization logs above for required setup fixes.")
            print("===================================================")

if __name__ == "__main__":
    run_pipeline()
