# IMPORTANT: 
# Before running this demo, you should fill your openai api-key in .env. If you can't find a file named '.env', make a copy
# of '.env.template' and rename it to '.env' 
import os
import numpy as np
from loguru import logger
import transformers
from transformers import  AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from PIL import Image
from uuid import uuid1
from pathlib import Path


assert transformers.__version__ >= "4.35.3"
MAX_NEW_TOKENS = 200
IMG_ROOT_PATH = "data/"

def generate_response(image: Image.Image, message: str, **kwargs):
    global model, processor
    prompt = f"USER: <image>\n{message}\nASSISTANT:"
    logger.info(f' ==== prompt ====\n{message}')
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    texts = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    responses = [{"content": text.split("ASSISTANT:")[-1].strip(), "index": i} for i,text in enumerate(texts)]
    logger.info(f' ==== responses ====\n{responses}')
    return responses


def generate_output(user_input: str, history: list, img: np.ndarray) -> str:
    global model, processor
    img_id = str(uuid1())
    img_save_pth = Path(f'data/{img_id}.jpeg')
    img_obj = Image.fromarray(img)
    img_obj.save(img_save_pth)
    # generate image description
    query = "Describe the image in detail."
    responses = generate_response(img_obj, query, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7)
    img_desc = responses[0]['content']

    # find related image information from collection
    docs = collection.similarity_search(query=img_desc, k=1)
    logger.info(f' ==== find context from collection ====\n{docs}')
    if len(docs)>0:
        related_img_info_lst = [
            f"description:\n{doc.page_content}\npath:\n{doc.metadata['path']}"
            for doc in docs
        ]
        related_img_info = "\n".join(related_img_info_lst)
    else:
        related_img_info = "No related image found."
    
    # add image description to collection
    collection.add_texts(texts=[img_desc], metadatas=[{'path':str(img_save_pth.absolute())}])
    logger.debug(f' ==== add texts to vector storage ====\n{img_desc}')
    
    # Visual question answer
    responses = generate_response(img_obj, user_input, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7)
    response = responses[0]['content']

    # Return model output.
    return f"**Related Image Infomation**\n{related_img_info}\n**Response**\n{response}"


# Define the ChatInterface, customize, and launch!
demo = gr.ChatInterface(
        generate_output,
        chatbot=gr.Chatbot(
            label="Chat with me!",
        ),
        textbox=gr.Textbox(
            placeholder="Message ...",
            scale=7,
            info="Input your textual response in the text field and your image below!"
        ),
        additional_inputs="image",
        additional_inputs_accordion=gr.Accordion(
            open=True,
        ),
        title="Language-Image Question Answering with bczhou/TinyLLaVA-v1-hf!",
        description="""
        This simple gradio app internally uses a Large Language-Vision Model (LLVM) and the Chroma vector database for memory.
        Note: this minimal app requires both an image and a text-based query before the chatbot system can respond.
        """,
        submit_btn="Submit",
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
)
if __name__ == '__main__':
    load_dotenv()
    
    model_id = "bczhou/tiny-llava-v1-hf"
    # model_id = "bczhou/TinyLLaVA-3.1B"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map = 'cuda'
    ).eval()# type:ignore
    processor = AutoProcessor.from_pretrained(model_id)
    # Use OpenAI's embeddings for our Chroma collection.
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),# type: ignore
    )
    # This generate a persistant collection. If you want to clear all cached information, delete folder 'chroma' in the same directory.
    collection = Chroma("conversation_memory", embeddings, persist_directory='chroma')
    # img_data = requests.get("https://imgur.com/Ca6gjuf.png").content
    # with open('sample_image.png', 'wb') as handler:
    #     handler.write(img_data)
    os.makedirs(IMG_ROOT_PATH, exist_ok=True)
    demo.queue().launch(server_port=7860)