import warnings
import os
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pypdf import PdfReader
import streamlit as st

# Replace with your actual key
GOOGLE_API_KEY = "AIzaSyAbu3uErPECsNdKxJ6AIEQ1UXmGwG1WlCI"  # Replace with your actual API Key
genai.configure(api_key=GOOGLE_API_KEY)

warnings.filterwarnings("ignore")
try:
    client = chromadb.Client(tenant="default_tenant")
except Exception as e:
    st.error(f"Error connecting to Chroma client: {e}")
    client = None

model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    return text_splitter.split_text(text)

def save_db_to_disk(db, db_path):
    db.persist(directory=db_path)
    print(f"Vector store saved to {db_path}")

def load_db_from_disk(db_path):
    if os.path.exists(db_path):
        db = Chroma.load(directory=db_path)
        print(f"Loaded vector store from {db_path}")
        return db
    return None

def process_documents(pdf_directory, db_path, existing_db=None):
    if existing_db:
        return existing_db

    # Try loading from disk if not passed
    db = load_db_from_disk(db_path)
    if db:
        return db

    all_texts = []
    metadata = []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = split_text_into_chunks(text)
                all_texts.extend(chunks)
                metadata.extend([{"source": pdf_path} for _ in chunks])

    if not all_texts:
        return None

    # Create and save the vector store to disk
    db = Chroma.from_texts(all_texts, embeddings, metadatas=metadata)
    save_db_to_disk(db, db_path)
    return db

# Directory where your PDFs are stored
pdf_directory = "B"  # Make sure this is the correct path. Create this directory and put your PDF files in it.
db_path = "chroma_db"  # Directory to store the vector database

# Using the db from a previous run if available, or creating a new one.
db = process_documents(pdf_directory, db_path, existing_db=None)

if db is None:
    st.error("No valid PDF files found in the directory. Please add PDF files to the /content/ directory.")
else:
    vector_index = db.as_retriever(search_kwargs={"k": 20})

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )

    def generate_response(full_query): 
        try:
            result = qa_chain({"query": full_query}) 
            explanation = result["result"]
            return explanation
        except Exception as e:
            return f"An error occurred: {e}"

    # Streamlit setup
    st.set_page_config(page_title="University Chatbot")
    with st.sidebar:
        st.title('University Chatbot')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if input := st.chat_input("Enter your question about the PDFs..."):
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        with st.chat_message("assistant"):
            with st.spinner("Searching for answers..."):
                # 1. Construct prompt with history
                conversation_history = ""
                for message in st.session_state.messages[-5:]:  # Include last 5 messages for context (adjust as needed)
                    conversation_history += f"{message['role']}: {message['content']}\n"

                full_query = f"{conversation_history}User: {input}" # include user query in prompt

                response = generate_response(full_query)  # Pass the full query with context
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
