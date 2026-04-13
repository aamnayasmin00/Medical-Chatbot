# Medical-Chatbot
This Medical Chatbot is a Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Streamlit. It uses a Qwen 2.5 7B model to provide context-aware answers grounded in a PDF knowledge base. Designed for accuracy, it efficiently retrieves medical data to minimize AI hallucinations.
md
💊 Medical Chatbot (RAG System)
A professional-grade Medical Chatbot built using LangChain, FAISS, and Streamlit. This system uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers to medical questions by referencing a local PDF-based knowledge library.

🚀 Key Features
PDF Knowledge Injection: Automatically processes and chunks medical PDF documents for searchable memory.

Vector Search: Uses FAISS for efficient similarity searching to find the most relevant context for any user query.

RAG Architecture: Combines retrieved context with the Qwen 2.5 7B model via the Hugging Face Inference Router to minimize hallucinations.

Streamlit UI: A clean, interactive chat interface for seamless user communication.

Persistent Memory: Maintains session-based chat history for a continuous conversation experience.

🛠️ Tech Stack
LLM: Qwen/Qwen2.5-7B-Instruct (via Hugging Face Router)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Orchestration: LangChain (LCEL)

Database: FAISS (Facebook AI Similarity Search)

Frontend: Streamlit


Python Version: 3.13 

📂 Project Structure
create_memory_for_llm.py: Scripts for loading PDFs, creating text chunks, and saving the FAISS vector database.

connect_memory_with_llm.py: A command-line implementation of the RAG pipeline.

medibot.py: The main Streamlit application providing the chatbot interface.

vectorstore/db_faiss: Directory where the searchable knowledge base is stored.

⚙️ Installation & Setup
1. Requirements
Ensure you have Python 3.13 installed. Install the necessary libraries (LangChain, Streamlit, FAISS, etc.).

2. Environment Variables
Create a .env file in the root directory and add your Hugging Face token:

Code snippet
HF_TOKEN=your_huggingface_token_here
3. Build the Knowledge Base
Place your medical PDFs in a folder named data/ and run the memory creation script:

Bash
python create_memory_for_llm.py
4. Run the Chatbot
Launch the Streamlit interface:

Bash
streamlit run medibot.py
