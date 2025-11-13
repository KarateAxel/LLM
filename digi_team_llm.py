import os
#Helps to get the user name
import getpass

from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.tools import tool


from langchain_ollama import OllamaEmbeddings

import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")


from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain.memory import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory


""" import opik
opik.configure(use_local=False) """
from opik.integrations.langchain import OpikTracer
os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"
opik_tracer = OpikTracer(
    project_name="Test-Project",
    tags=["digi_team_llm"],
)

import warnings
warnings.filterwarnings("ignore")

# auswahl des LLM
base_url = "http://localhost:11434"
#model = 'llama3.2:3b'
#model = 'llama3.3'
#model = 'qwen2.5:32b'
#model = 'qwen2.5:14b'
model = 'gpt-oss:20b'
llm = ChatOllama(model= model, base_url=base_url)
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=base_url)

#memory = ChatMessageHistory(session_id=getpass.getuser())
memory = ChatMessageHistory(session_id="test-session")

db_dp = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\transkript_professor"
vector_store = FAISS.load_local(db_dp, embeddings, allow_dangerous_deserialization=True)
retriever_ts_prof = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_dp = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\datenschutz_gesetze"
vector_store = FAISS.load_local(db_dp, embeddings, allow_dangerous_deserialization=True)
retriever_dp = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_swtv = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\swt_validation"
vector_store = FAISS.load_local(db_swtv, embeddings, allow_dangerous_deserialization=True)
retriever_swtv = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_qm = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\qualitaetsmanagement"
vector_store = FAISS.load_local(db_qm, embeddings, allow_dangerous_deserialization=True)
retriever_qm = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_zuMed = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\zulassung_von_MED"
vector_store = FAISS.load_local(db_zuMed, embeddings, allow_dangerous_deserialization=True)
retriever_zuMed = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

db_labeling = r"C:\Users\Digi-Team\LANGCHAIN-AND-OLLAMA-MAIN\labeling"
vector_store = FAISS.load_local(db_labeling, embeddings, allow_dangerous_deserialization=True)
retriever_labeling = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})

@tool
def ts_prof_search(query: str) -> str:
    """Search for information about robotic and automation.
    For any questions about robotic and automation from a professor, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_ts_prof.invoke(query)
    return response


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

@tool
def quality_management_search(query: str) -> str:
    """Search for information about quality managemen, validation and verification.
    For any questions about quality managementsystems, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_qm.invoke(query)
    return response

@tool
def zulassung_med_search(query: str) -> str:
    """Search for information about software tool validation and verification.
    For any questions about software validation and 21CFR Part 11, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_zuMed.invoke(query)
    return response

@tool
def labeling_search(query: str) -> str:
    """Search for information about software tool validation and verification.
    For any questions about software validation and 21CFR Part 11, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever_labeling.invoke(query)
    return response

tools = [ts_prof_search, dsgvo_search, swt_validation_search,quality_management_search,zulassung_med_search, labeling_search]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)



""" def chat_with_llm(user_id, question):
    for output in agent_with_chat_history.stream({"input": question},config={"configurable": {"session_id": user_id}},):
        if "output" in output:
            yield output["output"] """


def chat_with_llm(user_id, question):

    for output in agent_with_chat_history.stream({"input": question},config={"configurable": {"session_id": user_id}, "callbacks": [opik_tracer]}):
        if "output" in output:
            yield output["output"]


#print(chat_with_llm("test-session", "What ist the scope of the 21CFR Part 11"))

def chat_with_llm_test(user_id, question):
    output = agent_with_chat_history.invoke({"input": question},config={"configurable": {"session_id": "<foo>"}, "callbacks": [opik_tracer]},callbacks=[opik_tracer])
    return output['output']
    


#print(chat_with_llm_test("test-session", "Fasse mir das Interview mit dem Professor strukturiert zusammen."))