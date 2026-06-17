# pdf_bot_demo

This project requires three core libraries/concepts working together:

Document Loader: To read PDF files.
Vector Store & Embeddings: To convert text into searchable numerical meaning (the "Context").
LLM Chain: To take the retrieved context and formulate a final, accurate answer (The Prompt).

git clone https://github.com/haidong1965/pdf_bot_demo

cd pdf_bot_demo

pip install chromadb google-genai langchain_community pypdf

pip install  sentence-transformers

python -m venv my_rag_project_venv

my_rag_project_venv\Scripts\activate.bat
