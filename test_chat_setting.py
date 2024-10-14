import chainlit as cl

@cl.step
async def my_step():
    current_step = cl.context.current_step

    # Override the input of the step
    current_step.input = "My custom input"

    # Override the output of the step
    current_step.output = "My custom output"
