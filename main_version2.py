# Required imports
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from pathlib import Path

# Load environment variables
load_dotenv()
OpenAI_API = os.getenv("OPENAI_API_KEY")
ASTRA_DB_API = os.getenv("ASTRA_DB_API")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# Initialize Cassandra connection
cassio.init(token=ASTRA_DB_API, database_id=ASTRA_DB_ID)

# Set up Langchain and embeddings
llm = OpenAI(openai_api_key=OpenAI_API)
embedding = OpenAIEmbeddings(openai_api_key=OpenAI_API)
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="Bacuti",
    session=None,
    keyspace=None,
)

# Function to read all PDFs in ./reports folder
def load_all_reports():
    raw_text = ''
    reports_folder = Path("./reports")
    for pdf_file in reports_folder.glob("*.pdf"):
        pdf_reader = PdfReader(str(pdf_file))
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

# Function to process a single PDF file and add it to the vector store
def process_pdf_file(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Add texts to Cassandra vector store
    astra_vector_store.add_texts(texts)

# Initial setup: Load and store all reports in the ./reports folder
raw_text = load_all_reports()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)
astra_vector_store.add_texts(texts)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Streamlit frontend
st.title("Document Q&A System")
st.write("Enter a question below to query the documents.")

# File upload for new reports
st.subheader("Upload New Report")
uploaded_file = st.file_uploader("Drag and drop or select a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        process_pdf_file(uploaded_file)
        st.success("File processed and added to the database!")

# User input for the question
question = st.text_input("Ask a question:")
question = f"""
    You are an AI specializing in generating sustainability reports. Based on the given context, provide a detailed 
    Table of Contents and Executive Summary for Tata Motors (or the specified company) on sustainability initiatives. 
    Use topics relevant to sustainability like energy usage, waste reduction, and environmental impact. If context is 
    insufficient, respond with 'Information not available.'

    Question: {question}
"""

if question:
    with st.spinner("Processing your question..."):
        answer = astra_vector_index.query(question, llm=llm).strip()
        if answer:
            st.success("Answer:")
            st.write(answer)
        else:
            st.error("No answer found. Try rephrasing your question.")
