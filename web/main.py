import gradio as gr
import asyncio
from ollama import AsyncClient
import ollama
from conversation import (default_conversation, conv_templates,
                                   SeparatorStyle, LOGDIR)
from utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import time
import requests
import hashlib
import argparse
import datetime
import json
import os
import time
import base64

logger = build_logger("gradio_web_server", "gradio_web_server.log")

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

from ollama import Client
client = Client(host='http://localhost:11435')

headers = {"User-Agent": "HubFusion"}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledgebase.provider.pgvector import PgVectorInterface

from dotenv import load_dotenv

load_dotenv() 

DB_CONFIG = {
            "host": os.getenv("VECTOR_DB_HOST"),
            "port": os.getenv("VECTOR_DB_PORT"),
            "database": os.getenv("VECTOR_DB_DATABASE"),
            "user": os.getenv("VECTOR_DB_USER"),
            "password": os.getenv("VECTOR_DB_PASSWORD")
        }


pgVector = PgVectorInterface(DB_CONFIG)

from transformers import AutoTokenizer, AutoModel
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def getEmbedding(document, tokenizer, model):
    inputs = tokenizer(document, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool the token embeddings to create a single vector representation of the document
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    return embedding.tolist()



settings = {'model': 'llava'}

# Define the chat prediction function (example using Ollama)
async def predict(message, history, system_prompt, tokens):

    global settings
   # formatted_messages2 = [{'role': 'user', 'content': item[0]}, {'role': 'assistant', 'content': item[1]} for item in history]
    if len(history) == 0:
        embedding = getEmbedding(message, tokenizer, model)
        context = pgVector.search(embedding, 1)[0]
        message = f"""
            Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {message}
            Answer: 
            """
    
    print("history", history)
    formatted_messages2 = []
    for i, item in enumerate(history):
        formatted_messages2.append({'role': 'user', 'content': item[0]})
        formatted_messages2.append({'role': 'assistant', 'content': item[1]})
    formatted_messages2 += [{'role': 'user', 'content': message}]
    
    '''
    history_transformer_format = history + [[message, ""]]

    # Format the message for the chat
    formatted_messages = [{'role': 'user', 'content': message}] + [
        {'role': 'assistant', 'content': item[1]} for item in history_transformer_format
    ]
    
        

        
    print("message:", message)
    print(history)
    print(formatted_messages2)
   # print("history:", history)
   # print("history_transformer_format:", history_transformer_format)
   # print("formatted_messages:", formatted_messages)

    '''
    # Create an async client and start streaming response
    partial_message = ""
    async for part in await AsyncClient(host='http://localhost:11435').chat(model=settings['model'], messages=formatted_messages2, stream=True):
        new_token = part['message']['content']
        if new_token != '<':  # Continue accumulating message content
            partial_message += new_token
            
            yield partial_message

CSS = """
#component-4 {
    height: 300px;
}
"""



def add_text_(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    
    if image is not None:
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    else:
        state = default_conversation.copy()
        
    state.append_message(state.roles[0], text)
    state.append_message("assistant", None)
  #  state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5
    #[state, chatbot, textbox, imagebox] + btn_list

async def gen(state, temperature, top_p, max_new_tokens, request: gr.Request):
    model='llava'
    

    #print(state.messages)
    #[['USER', ('Опиши как создать такого же снеговика?', <PIL.Image.Image image mode=RGB size=1122x748 at 0x7F8EA3BFF7C0>, 'Default')]]
    formattedMessages= state.getOllamaFormattedMessages()
    #print(formattedMessages)
    partial_message = ""
    async for part in await AsyncClient(host='http://localhost:11435').chat(model='llava', messages=formattedMessages, stream=True):
        new_token = part['message']['content']
        if new_token != '<':  # Continue accumulating message content
            partial_message += new_token
    
            state.messages[-1][-1] = partial_message
            
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            
            
            

def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    
    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

    
    
def generate(state, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    
    #model_name = model_selector
    model_name = 'llava'

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

title_markdown = ("""
<img src="https://hubfusion.pro/wp-content/uploads/2024/08/HubFusion_light-1-1024x244.png" alt="HubFusion" width="150" height="120"/>

[[Project Page](https://hubfusion.pro)]
[[Code](https://github.com/braincxx/HubFusion)] 
""")

def build_interface(cur_dir=None, concurrency_count=10):
    
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    
    
    with gr.Blocks(css="div#component-4.block.svelte-12cmxck {height: 500px;}", fill_height=True) as demo:
        gr.Markdown(title_markdown)
        state = gr.State()
        # Create tabs
        with gr.Tab("Чат", elem_id="tab-chat"):
            '''
            embedding = getEmbedding(message, tokenizer, model)
            context = pgVector.search(embedding, 1)[0]
            message = f"""
                Context information is below.
                ---------------------
                {context}
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: {message}
                Answer: 
                """
            '''
            #system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
            #slider = gr.Slider(10, 100, render=False)
            
            #chat_interface = gr.ChatInterface(fn=predict, fill_height=True)
           # with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
            with gr.Row():
                with gr.Column(scale=3):
                    

                    imagebox = gr.Image(type="pil")
                    image_process_mode = gr.Radio(
                        ["Crop", "Resize", "Pad", "Default"],
                        value="Default",
                        label="Preprocess for non-square image", visible=False)

                    if cur_dir is None:
                        cur_dir = os.path.dirname(os.path.abspath(__file__))
                    gr.Examples(examples=[
                        [f"{cur_dir}/examples/snow.jpg", "Опиши как создать такого же снеговика?"],
                        [f"{cur_dir}/examples/bereza.jpg", "Дай информацию о дереве на изображении"],
                    ], inputs=[imagebox, textbox])

                    with gr.Accordion("Parameters", open=True) as parameter_row:
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                        max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            label="Чат",
                            height=650,
                            layout="panel",
                    )
                    with gr.Row():
                        with gr.Column(scale=8):
                            textbox.render()
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Send", variant="primary")
                    with gr.Row(elem_id="buttons") as button_row:
                        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                        #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                        clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

           

        with gr.Tab("База знаний"):
            def process_input(text, file):
                if file is not None:
                    text = file.decode('utf-8')  # Ensure text is properly decoded
                    
                embedding = getEmbedding(text, tokenizer, model)
                pgVector.insert(text, embedding)
                return "Знания добавлены!"

            with gr.Row():
                file_input = gr.File(label="Загрузите текстовый файл", file_types=[".txt"], type="binary", interactive=True)
                line_input = gr.Textbox(label="Введите текст")
            
            with gr.Row():
                submit_button = gr.Button("Загрузить в базу знаний")
            
            with gr.Row():
                text_output = gr.Textbox(label="", interactive=False, visible=True)
            
            def submit_action(text, file):
                # Call process_input function to handle the text and file
                result = process_input(text, file)
                
                # Clear the input fields
                return "", None, result
            
            # On button click, process either the uploaded file or the manual input
            submit_button.click(submit_action, inputs=[line_input, file_input], outputs=[line_input, file_input, text_output])

        # Tab to select model
        with gr.Tab("Настройки"):
            model_choice_dropdown = gr.Dropdown(
                choices=["llava", "mistral-nemo", "codegemma:2b"], 
                label="Выберите модель"
            )

            # Output for displaying updated model information
            model_output = gr.Textbox(label="Selected Model", visible=False)

            # Function to update the global dictionary and display the selected model
            def update_dict(selected_value):
                global settings
                # Store the selected value in the dictionary
                settings["model"] = selected_value
                return f"Model {selected_value} selected."
            
            # Link the dropdown change to the function that updates the dictionary
            model_choice_dropdown.change(fn=update_dict, inputs=model_choice_dropdown, outputs=model_output)
         
        #events
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        submit_btn.click(
                add_text_,
                [state, textbox, imagebox, image_process_mode],
                [state, chatbot, textbox, imagebox] + btn_list
            ).then(
                gen,
                [state, temperature, top_p, max_output_tokens],
                [state, chatbot] + btn_list,
                concurrency_limit=concurrency_count
            )
            
    return demo

# Run the Gradio app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(allowed_paths=["/"])
    