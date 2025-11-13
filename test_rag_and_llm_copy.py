import os

from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import (
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                        ChatPromptTemplate,
                                        MessagesPlaceholder
                                        )

from langchain_core.tools import tool


from langchain_community.tools import TavilySearchResults

from langchain_ollama import OllamaEmbeddings

import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain import hub

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from langchain import hub

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

llm = ChatOllama(model='qwen2.5:32b', base_url='http://localhost:11434')

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

@tool
def web_search(query: str) -> str:
    """
    Search the web for latest information about software verification and validation .
    for examples, software quality assurance, software testing etc.
    
    Args:
    query: The search query
    """
    
    search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )
    response = search.invoke(query)
    return response


db_dp = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\datenschutz_gesetze"
vector_store = FAISS.load_local(db_dp, embeddings, allow_dangerous_deserialization=True)

retriever_dp = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_swtv = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\datenschutz_gesetze"
vector_store = FAISS.load_local(db_swtv, embeddings, allow_dangerous_deserialization=True)

retriever_swtv = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

@tool
def dsgvo_search(query: str) -> str:
    """Search for information about data protection and AI regulation.
    For any questions about data protection and AI regulation, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_dp.invoke(query)
    return response

@tool
def swt_validation_search(query: str) -> str:
    """Search for information about software tool validation and verification.
    For any questions about software validation and 21CFR Part 11, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_swtv.invoke(query)
    return response


#print(dsgvo_search("Welche Praktiken im KI-Bereich sind verboten?"))

#Promt Tamplate sollte angepasst und geändert werden
prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [web_search, dsgvo_search]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

runnable_with_history = RunnableWithMessageHistory(agent_executor, get_session_history, 
                                                   input_messages_key='input', 
                                                   history_messages_key='history')

runnable_with_history = RunnableWithMessageHistory(agent_executor, get_session_history)

""" question = "Welche Praktiken im KI-Bereich sind verboten?"
question = "Wie kann ich am besten eine RPA Software für die Software validierung im Medezinbereich, validieren"
response = agent_executor.invoke({'input': question})

print(response['output']) """

def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke(
        {'input': input},
        config={'configurable': {'session_id': session_id}}
    )

    return output


question = "Meine Name ist Alex, wie geht es dir?"
user_id = "laxmi_kant"
chat_with_llm(user_id, question)