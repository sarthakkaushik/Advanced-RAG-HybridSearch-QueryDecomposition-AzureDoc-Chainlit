import chainlit as cl
from chainlit.input_widget import TextInput


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            TextInput(id="AgentName", label="Agent Name", initial="AI"),
        ]
    ).send()
    value = settings["AgentName"]