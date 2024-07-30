import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from constants import *

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY2')
pinecone_key = os.getenv('PINECONE_API_KEY')

# Prep documents to be uploaded to the vector database (Pinecone)

# Split documents into smaller chunks

# Choose the embedding model and vector store 
