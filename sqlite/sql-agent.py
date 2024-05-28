from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3",
)

## Markdown pip install markdown && !pip install unstructured > /dev/null

from langchain_community.document_loaders import UnstructuredMarkdownLoader
markdown_path = "./sqlite/database-structure.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()

## -------------- part3 --------------

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


## -------------- part4 --------------
## pip install chromadb

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)


question="Que dados posso obter da table Artist?"
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

while True:
    question = input("Enter your question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break

    response = qachain.invoke({"query": question})
    print(response)
    print()