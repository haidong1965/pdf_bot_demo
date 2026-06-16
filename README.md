# pdf_bot_demo

This project requires three core libraries/concepts working together:

Document Loader: To read PDF files.
Vector Store & Embeddings: To convert text into searchable numerical meaning (the "Context").
LLM Chain: To take the retrieved context and formulate a final, accurate answer (The Prompt).

pip install chromadb sentence-transformers google-genai langchain_community pypdf
