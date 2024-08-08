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
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


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


# Load the document, split it into chunks, embed each chunk, load into the vector store, and perform hybrid search
# Documentation here: https://python.langchain.com/v0.1/docs/integrations/vectorstores/azuresearch/; https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/

azure_endpoint: str = "" #Azure OpenAI endpoint
azure_openai_api_key: str = "" #API key
azure_openai_api_version: str = "2023-05-13"
azure_deployment: str = "text-embedding"

vector_store_address: str = "" #Azure search endpoint
vector_store_password: str = "" #Azure search admin key

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

index_name: str = "maas-chatbot-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

loader = TextLoader('r'C:\Users\kimjoh\Downloads\azureaidoc\document.txt').load()

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vector_store.add_documents(documents=docs)

docs = vector_store.similarity_search(
    query="How do I disable content filters for Azure AI Serverless endpoints?",
    k=3,
    search_type="hybrid",
)
print(docs[0].page_content)
