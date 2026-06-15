from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- 1. Load Documents ---
pdf_directory = "./pdf/" # Folder containing all your PDFs
loader = PyPDFLoader(pdf_directory)
documents = loader.load()

print(f"Successfully loaded {len(documents)} pages.")

# --- 2. Split into Chunks (Crucial Step!) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # How many tokens per chunk
    chunk_overlap=200 # How much overlap to ensure context isn't lost between chunks
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} manageable chunks.")


# --- 3. Create Embeddings & Store (The Index) ---
# OpenAIEmbeddings converts text chunks into vectors using AI math.
embeddings = OpenAIEmbeddings() 

# ChromaDB is the local vector database that stores and indexes these vectors.
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db" # Saves your index locally
)

print("? Indexing complete! Your knowledge base is ready.")

