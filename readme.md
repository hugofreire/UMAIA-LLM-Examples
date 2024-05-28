## 1. Instalar Ollama : https://ollama.com/

## 2. Descarregar o Modelo llama3 e Model de Embeddings
```bash
ollama pull ollama run llama3
ollama pull mxbai-embed-large
```

## 2. Inicializar Ambiente Virtual e Bibliotecas

```bash
python3 -m venv myenv
source myenv/bin/activate

pip install langchain_community
pip install bs4
pip install chromadb

```


## 3. Importar Ollama + Inicializar Ollama e Executar primeiro Prompt

```python
from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3",
)

print(ollama.invoke("Porquê que o ceu é azul?"))

```

## 4. Adicionar document_loader + text_splitter + vectorstore

```python

# Primeiro, é necessário instalar a biblioteca 'bs4' com: pip install bs4

from langchain_community.document_loaders import WebBaseLoader

# Carrega a página da web especificada
loader = WebBaseLoader("https://www.eventbrite.pt/e/convergencia-2024-comunicacao-media-arte-e-tecnologia-tickets-913778154057")
data = loader.load()

```

## 5. Divide o texto carregado em partes menores
```python

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configura o divisor de texto para criar segmentos de até 4000 caracteres com uma sobreposição de 1000 caracteres
text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
all_splits = text_splitter.split_documents(data)

```

## 6. Criar embeddings e armazenar vetores

```python
# É necessário instalar o ChromaDB com: pip install chromadb

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Cria embeddings para os documentos usando o modelo especificado
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")

# Armazena os documentos divididos em um repositório de vetores
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
```

## 7. Realizar uma busca por similaridade
```python
# Define a pergunta a ser feita
question = "Há algum convidado chamado Hugo Freire?"

# Realiza a busca por documentos similares à pergunta
docs = vectorstore.similarity_search(question)

# Imprime a quantidade de documentos encontrados
for idx, doc in enumerate(docs):
    print(f"Document {idx + 1}:")
    print(doc.page_content)
    print("\n" + "-"*50 + "\n")
```

## 8. Criar uma cadeia de QA e obter a resposta
```python
from langchain.chains import RetrievalQA
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
response = qachain.invoke({"query": question})

print(response)
```