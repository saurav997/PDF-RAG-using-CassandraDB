from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
from datasets import load_dataset
from dotenv import load_dotenv
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

OpenAI_API = os.getenv("OPENAI_API_KEY")
ASTRA_DB_API = os.getenv("ASTRA_DB_API")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
# do this for each report within ./reports folder 
pdfreader = PdfReader("report")

raw_text = ''
for i,page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

cassio.init(token = ASTRA_DB_API, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key = OpenAI_API)
embedding = OpenAIEmbeddings(openai_api_key = OpenAI_API)
astra_vector_store = Cassandra(
    embedding = embedding,
    table_name = "Bacuti",
    session = None,
    keyspace = None,
)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
astra_vector_store.add_texts(texts)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

#streamlit frontend that gets querry goes here:

question = ""
answer = astra_vector_index.query(question,llm = llm).strip()

#streamlit frontend for displaying the response in a neatly parsed format