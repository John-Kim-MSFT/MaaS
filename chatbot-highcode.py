# %pip install langchain-chroma
# %pip install -U langchain-openai
# %pip install --upgrade pip

import openai
import json
import os
from langchain_openai import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


# Configuring API from a config file to protect sensitive information
with open(r'config1.json') as config_file:
    config_details = json.load(config_file)
openai_api_base=config_details['OPENAI_API_BASE']
openai_api_version=config_details['OPENAI_API_VERSION']
deployment_name=config_details['DEPLOYMENT_NAME']
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_type="azure"

#Instantiation of AzureChatOpenAI based on LangChain framework
llm = AzureChatOpenAI(
    azure_openai_api_key=openai_api_key,
    azure_deployment_name=deployment_name,
    azure_api_base=openai_api_base,
    azure_api_version=openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# Load the document, split it into chunks, embed each chunk and load it into the vector store. Documentation here: https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
raw_documents = TextLoader('r'C:\Users\kimjoh\Downloads\azureaidoc\document.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # Chunking based on size is not the best way to improve RAG -- one idea is to create embeddings of summaries of chunks; the low code solution uses semantic + hybrid search to improve the query results
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "How do I disable content filters for Azure AI Serverless endpoints?"
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content) 

