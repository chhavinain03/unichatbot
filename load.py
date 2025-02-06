__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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

def process_documents(pdf_directory):
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

    db = Chroma.from_texts(all_texts, embeddings, metadatas=metadata)
    return db

pdf_directory = "B"  # Make sure this is the correct path. Create this directory and put your pdf files in it.
db = process_documents(pdf_directory)

if db is None:
    st.error("No valid PDF files found in the directory. Please add PDF files to the /content/ directory.")
else:
    vector_index = db.as_retriever(search_kwargs={"k": 10})

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )

    def generate_response(full_query): # changed input to full_query
        try:
            result = qa_chain({"query": full_query}) # pass the full query
            explanation = result["result"]
            return explanation
        except Exception as e:
            return f"An error occurred: {e}"

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
