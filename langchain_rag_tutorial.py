import os
#Lybrary für LLM
from langchain_ollama import ChatOllama 
# auswahl des LLM
base_url = "http://localhost:11434"
model = 'llama3.2:3b'
llm = ChatOllama(model= model, base_url=base_url)
#Lybray für Embedding Modell
from langchain_ollama import OllamaEmbeddings
#auswahl des Embedding Modells
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
#Lybrarys für die Vektor DB
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore
#Vektor generieren
embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)