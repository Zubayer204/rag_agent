from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from dotenv import load_dotenv

load_dotenv()

MEMORY_KEY = "chat_history"

with open("system_prompt.txt", "r") as file:
    system_prompt = file.read()

embeddings = VoyageAIEmbeddings(
    model="voyage-2"
)

# Create a Chroma vector store
docsearch = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Initialize message history for conversation
message_history = ChatMessageHistory()

# Memory for conversational context
memory = ConversationBufferMemory(
    memory_key=MEMORY_KEY,
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)


tools = [
    create_retriever_tool(
        retriever=docsearch.as_retriever(),
        name="policy_info_searcher",
        description="Searches information regarding insurance, products, benefits and all other travel related queries. Also contains info about human agent contact",
    )
]

prompt = ChatPromptTemplate.from_messages(  
    [
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

def create_openai_agent() -> AgentExecutor:
    llm_openai = ChatOpenAI(
        model_name="gpt-4o",
        streaming=True
    )

    llm_with_tools = llm_openai.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools,handle_parsing_errors=True)
