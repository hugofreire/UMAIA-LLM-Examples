import requests
from bs4 import BeautifulSoup
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# URL of the real estate listing
url = "https://www.imdb.com/title/tt1684562/?ref_=fea_em00063_3_title_sm"

loader = WebBaseLoader(url)
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)

# Embed the document chunks using Ollama embeddings
ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=ollama_embeddings)

# Create a RetrievalQA chain using Ollama
ollama = Ollama(base_url='http://localhost:11434', model="llama3")
retriever = vectorstore.as_retriever()
qachain = RetrievalQA.from_chain_type(ollama, retriever=retriever)

# Function to query the vector store and get answers
def get_answer(question):
    response = qachain.invoke({"query": question})
    return response['result']

# Define questions to extract specific data
questions = {
    "title": "What is the title of movie?",
    "director": "What director of the movie?",
    "Writers": "What the Writers?",
    "description": "whats the Storyline of the movie?"
}

# Extract the data by asking questions
extracted_data = {}
for key, question in questions.items():
    extracted_data[key] = get_answer(question)

# Format the extracted data into JSON
json_data = json.dumps(extracted_data, indent=4, ensure_ascii=False)
print(json_data)
