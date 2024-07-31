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
loader = DirectoryLoader('./', glob='**/*.pdf', loader_cls=PyPDFLoader)
raw_docs = loader.load()
# print(raw_docs)
# print(len(raw_docs))

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_docs)
print(f"Adding {len(documents)} to Pinecone!")

# Choose the embedding model and vector store 
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name=PINECONE_INDEX)
print('Documents uploaded to Pinecone')