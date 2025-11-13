#Lybrary wird benötigt um das Embedding Modell zu laden
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import (
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                        ChatPromptTemplate,
                                        MessagesPlaceholder
                                        )

#Lybrarys für die Vektor DB
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

#Lybrary für das RAG mit LLAMA 
from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain import hub

#Dotenv is a zero-dependency module that loads environment variables from a .env file. For monitoring.
from dotenv import load_dotenv
load_dotenv()

#For chat hisory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

#Ignoriert eine Warnmeldung die bei zwei verschiedenen installierten Vektor DBs auftauchen kann
import warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# auswahl des LLM
base_url = "http://localhost:11434"
#model = 'llama3.2:3b'
#model = 'llama3.3'
model = 'qwen2.5:32b'
llm = ChatOllama(model= model, base_url=base_url)

#auswahl des Embedding Modells
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
#auswahl der Vektor DB
#db_name = r"Industie_Normen"
db_name = r"swt_validation"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

#Promt Tamplate
prompt = """
    Sie sind ein Assistent für die Beantwortung von Fragen. Beantworten Sie die Frage mit Hilfe der folgenden Kontextinformationen.
    Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen.
    Achten Sie darauf, dass sich Ihre Antwort auf die Frage bezieht und dass sie nur aus dem Kontext heraus beantwortet wird. Gib bitte wenn möglich den dazugehörigen Artikel oder Absatz an, unter dem die Information gefunden wurde. Antworte bitte auf deutsch.
    Question: {question} 
    Context: {context} 
    Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt)

#Sucht in der DB nach passenden Chunks
retriever = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})


#Funktion wird aus der rag_chain aufgerufen, fügt die Inhalte zusammen und übergibt sie als centext zurück an die Chain
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])



#Initialisierung der RAG-Chain
rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Runnable with History


def ask_llm(question):
    docs = retriever.invoke(question)
    return rag_chain.invoke(question)

#question = "Ist es ein Datenschutzvergehen, wenn ein Skript einem Mitarbeiter Mails sendet?"
#question = "Welche Praktiken im KI-Bereich sind verboten?"
question = "What ist the scope of the 21CFR Part 11"
print(ask_llm(question))