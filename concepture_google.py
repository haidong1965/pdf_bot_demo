# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer 
import os
from pathlib import Path
import time 
import chromadb # Make sure to import the library

from dotenv import load_dotenv
from google import genai # <-- Import the correct library

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError 

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ['HF_TOKEN'] ="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"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

try:
    from langchain_community.document_loaders import PyPDFLoader # Loader for PDFs
except ImportError as e:
    print(f"?? CRITICAL LIBRARY ERROR: Failed to load PDF loader. Did you reinstall dependencies? Error details: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter 
except ImportError:
    print("?? WARNING: Could not find 'langchain_text_splitters'. Falling back to explicit core import.")
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter # Fallback attempt 1
    except ImportError as e:
         print(f"?? CRITICAL FAILURE: Cannot locate Text Splitter class in any known path. Error: {e}")

from chromadb import Client 

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

PDF_DIR = Path(r"C:\python_apps\pdf_bot_demo\pdf") # <<< ?? CHANGE THIS PATH! ??

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
    system_prompt = f"""
    You are an expert Q&A assistant. Your sole task is to answer user questions based ONLY on the context provided below.
    If the context does not contain the information required to answer the question, you must reply with: "I do not have enough information in the provided documents to answer that."
    Do not use external knowledge.

    --- CONTEXT ---
    {context_text}
    """
    gemini_client = genai.Client(api_key=API_KEY) 

    # --- APPLYING RETRY LOGIC HERE ---
    MAX_ATTEMPTS = 5
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            chat = gemini_client.chats.create(model='gemini-2.5-flash') 
            full_user_message = f"Question: {user_query}"
            
            # Attempt to send the message
            response = chat.send_message(full_user_message, config={"system_instruction": system_prompt})
            return response.text # SUCCESS! Exit and return answer immediately.

        except Exception as e:
            error_msg = str(e)
            # Check for common service unavailability indicators (like rate limits or server overload)
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "RATE_LIMIT" in error_msg:
                if attempt < MAX_ATTEMPTS:
                    wait_time = 2 ** attempt # Exponential backoff (2s, 4s, 8s...)
                    print(f"\n?? Service temporarily unavailable ({error_msg[:50]}...). Retrying in {wait_time} seconds... (Attempt {attempt}/{MAX_ATTEMPTS})")
                    time.sleep(wait_time)
                else:
                    # Failed after all retries
                    print("\n?? Maximum retry attempts reached for service unavailability.")
                    break # Exit the loop to fail gracefully below
            else:
                # This is a different, fatal error (like bad credentials or wrong model name). Do not retry.
                raise e 

    # If the loop finishes without returning, it means all retries failed due to service errors.
    return "I apologize, but the knowledge base connection is currently unstable. Please try running this process again in a few minutes."

def upsert_to_vectorstore(chunks, embeddings, metadatas):
    """Upserts document content and embeddings into ChromaDB using guaranteed unique IDs."""
    if not chunks or len(embeddings) == 0:
        print("?? Cannot upsert embeddings: No data available.")
        return False

    try:
        COLLECTION_NAME = "rag_knowledge_base" 
        client = chromadb.PersistentClient(path="./chroma_db")
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
    embeddings = None 
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
    
# ------------------------------------------------------------
# 1. RETRIEVE CONTEXT (The whole block)
# ------------------------------------------------------------

    # --- VARIABLES TO BE USED LATER (Define these before Phase 1) ---
    user_query = "What is the formula used to calculate the OH processing stage index stgIx?"  
    context = []
    collection = None # Initialize it to None first!
    query_list = [] # Initialize query_list here!
    
    # ========================================
    # PHASE 1: SETUP & CONNECTION (MUST RUN FIRST)
    # ========================================
    try:
        print("\n[PHASE 1/3] Establishing Database Connection...")
        client = chromadb.PersistentClient(path="./chroma_db") # Uses the folder INSIDE C:\ai\
        COLLECTION_NAME = "rag_knowledge_base" 

        # Attempt to get the collection object - This defines the 'collection' variable!
        collection = client.get_collection(name=COLLECTION_NAME)
        print("? Phase 1 Success: Database connection established.")
    
    except Exception as e:
        print(f"? FATAL SETUP ERROR in Phase 1: Could not connect to the database. Details: {e}")
        # If setup fails, we must exit gracefully before proceeding.
        exit() 

    # ========================================
    # PHASE 2: EMBEDDING GENERATION (BAAI BGE)
    # ========================================
    try:
        print("\n[PHASE 1.5/3] Loading Embedding Model & Generating Embeddings...")
        
        # 1. Load the specific BAAI model
        model_name = 'BAAI/bge-small-en-v1.5'
        embedding_model = SentenceTransformer(model_name)
        
        # 2. Encode the user query text into a vector list
        embeddings_result = embedding_model.encode(user_query)
        
        # ChromaDB expects the embeddings as a list of lists/arrays
        query_list = [embeddings_result] # We wrap it in a list to match expected format for .query()
        print("? Phase 1.5 Success: Embeddings generated and stored in 'query_list'.")

    except Exception as e:
        print(f"? ERROR during Phase 1.5 Embedding Generation: {e}")
        # If embedding fails, we cannot proceed to query.
        exit()

    # ========================================
    # PHASE 3: RETRIEVAL (Uses 'collection' from Phase 1)
    # ========================================
    if collection: # Check if the connection was successfully made in Phase 1
        try:
            print("\n[PHASE 2/3] Querying Knowledge Base...")
            
            # !!! IMPORTANT: Make sure query_list is defined somewhere above this block !!!
            results = collection.query(
                query_embeddings=query_list,  
                n_results=5,                       
                include=['documents', 'metadatas'] 
            )

            print("? Phase 2 Success: Context retrieved.")
            
            # Context Extraction Logic (Fixes the original NameError)
            retrieved_documents = results['documents']
            retrieved_metadatas = results['metadatas']

            if len(retrieved_documents) != len(retrieved_metadatas):
                print("?? Warning: Mismatch in retrieval results.")
                context = []
            else:
                context = []
                for doc_text, meta in zip(retrieved_documents, retrieved_metadatas):
                    context.append({'document': doc_text, 'metadatas': meta})

        except Exception as e:
            print(f"? ERROR during Phase 2 Retrieval: {e}")
            context = [] # Reset context on failure
    else:
        print("\n[PHASE 2/3] SKIPPED because database connection failed in Phase 1.")


    # ========================================
    # PHASE 4: ANSWER GENERATION (Uses 'context' from Phase 2)
    # ========================================
    if context: # Only run if we successfully gathered context data
        print("\n[PHASE 3/3] Generating Final Answer...")

        # Context simulates extracting specific process details from both sources
        mock_context = [
              {
                "document": "The index (stgIx) is calculated using the following formula: stgIx = (blkIx * 8) + stgOfs. Here, blkIx is the OH block index, and stgOfs is the processing stage offset",
                "metadatas": { "source_file": "Slides.pdf", "page_number": 401, "topic": "5.3.6 Table indexing"}
              }
        ]
    # 2. RUN THE LOGIC, passing the client object to the function
        try:
            final_answer = generate_answer(user_query, mock_context)
        except Exception as e:
            # If generating the answer fails completely (e.g., missing API key at the highest level)
            print(f"\n? FATAL ERROR in main execution loop: {e}")
            final_answer = "Could not generate an answer due to a critical system error."

        
        print("\n" + "="*50)
        print("? PIPELINE COMPLETE ?")
        print("? FINAL ANSWER:")
        print(final_answer)
        print("="*50)
    else:
        print("\n[PHASE 3/3] SKIPPED. Cannot generate an answer without context.")
