import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from constants import *

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY2')
pinecone_key = os.getenv('PINECONE_API_KEY')

# the prompt: we will be changing this soon
prompt = "hello world!"

# Note: we must use the same embedding model that we used when uploading the docs
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
document_vectorstore = PineconeVectorStore.from_documents(index_name=PINECONE_INDEX, embedding=embeddings)
retriever =document_vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0.7)
template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
# Querying the vector database for "relevant" docs then create a retriever

# create a context by using the retriever and getting the relevant docs based on the prompt

# show the thought process by looping over all relevant docs, showing the source and the content


# build a prompt template using the query and the context and build the prompt with context


# Asking the LLM for a response from our prompt with the provided context using ChatOpenAI and invoking it
# Then print the results content
