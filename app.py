import helpers
from langchain_core.messages import AIMessage, HumanMessage
import chainlit as cl


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "GlennAI"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def on_chat_start():
    # set starting message
    welcome_message = cl.Message(
        "Hi there! GlennAI here to help you with all your travel queries!"
    )

    await welcome_message.send()
    # create agent
    agent = helpers.create_openai_agent()

    # store the agent
    cl.user_session.set("agent", agent)

    # store the message history
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(input_message: cl.Message):
    chat_history = cl.user_session.get("chat_history")


    msg = cl.Message(content="")
    await msg.send()
        
    # Retrieve the chain from user session
    agent = cl.user_session.get("agent")     

    async for event in agent.astream_events(
        {"input": input_message.content, "chat_history": chat_history},
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                await msg.stream_token(content)
    
    # update the chat history
    chat_history.extend(
        [
            HumanMessage(content=input_message.content),
            AIMessage(content=msg.content),
        ]
    )

    # send the answer back to the user
    await msg.update()
