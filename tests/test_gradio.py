import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


import gradio as gr
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share=True, server_name=os.environ['HOST_NAME'])
