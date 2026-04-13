import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Your specific LCEL and Vectorstore libraries
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Page and Environment
st.set_page_config(page_title="Medical Chatbot", page_icon="💊")
st.title('Medical Chatbot 🤖')

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

# 2. Initialize Session State for Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 3. Load the "Memory" (FAISS) - Cached so it doesn't reload every click
@st.cache_resource
def load_rag_system():
    # Cloud-Based Embeddings (Matching your code)
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    
    # Path to your saved FAISS folder
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Database folder not found at {DB_FAISS_PATH}. Please run your 'Create Memory' script first.")
        return None

    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={'k': 3})

retriever = load_rag_system()

# 4. Setup the "Machine" (LLM + Prompt + Chain)
if retriever:
    # Your specific Model and Router
    HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_tokens=512
    )

    # Your specific Prompt Template
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know. 

    Context: {context}
    Question: {input}
    Answer directly:"""

    prompt_template = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    # Your LCEL Logic
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "input": RunnablePassthrough()}
        | prompt_template 
        | llm 
        | StrOutputParser()
    )

    # 5. Chat Input
    user_input = st.chat_input('Ask about medical symptoms or treatments...')

    if user_input:
        # Show User Message
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        
        # Generate and Show Response
        with st.chat_message('assistant'):
            with st.spinner("Searching memory and generating response..."):
                try:
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
                    # Store in state
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                except Exception as e:
                    st.error(f"Oops! Something went wrong: {str(e)}")