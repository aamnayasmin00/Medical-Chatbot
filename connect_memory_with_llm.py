import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

# 1. Setup the Model (Updated for 2026 Free Tier Stability)
# Qwen 2.5 is currently the most widely supported 'Chat' model on the free router.
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 

llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    model=HUGGINGFACE_REPO_ID,
    temperature=0.5,
    max_tokens=512
)

# 2. Rules for the AI
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. 

Context: {context}
Question: {input}
Answer directly:"""

prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

# 3. Load Database (Cloud-Based to avoid 15GB downloads)
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

# Load your local FAISS database
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3})

# 4. The Machine (LCEL Pipe Logic)
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
     "input": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# 5. Run
query = input("Write query here: ")
print(f"Routing request through Cloud Bridge using {HUGGINGFACE_REPO_ID}...")
print("\nRESULT:", rag_chain.invoke(query))