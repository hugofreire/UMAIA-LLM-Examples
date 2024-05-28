from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3",
)

print(ollama.invoke("Porquê que o ceu é azul?"))

## -------------- part2 --------------
## pip install bs4

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.eventbrite.pt/e/convergencia-2024-comunicacao-media-arte-e-tecnologia-tickets-913778154057")
data = loader.load()


## -------------- part3 --------------

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


## -------------- part4 --------------
## pip install chromadb

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)


question="De que se trata o evento?"
docs = vectorstore.similarity_search(question)

# Assuming the documents have a 'content' field with the text
for idx, doc in enumerate(docs):
    print(f"Document {idx + 1}:")
    print(doc.page_content)
    print("\n" + "-"*50 + "\n")



from langchain.chains import RetrievalQA
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
response = qachain.invoke({"query": question})

print(response)