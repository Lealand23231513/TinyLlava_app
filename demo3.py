import gradio as gr
import os
from loguru import logger
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from PIL import Image
from uuid import uuid3, NAMESPACE_DNS
from pathlib import Path
import shutil


MAX_NEW_TOKENS = 200
IMG_ROOT_PATH = "data/"
THRESHOLD = 0.01

load_dotenv()

model_id = "bczhou/tiny-llava-v1-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, low_cpu_mem_usage=True, device_map="cuda"
).eval()  # type:ignore
processor = AutoProcessor.from_pretrained(model_id)
# Use OpenAI's embeddings for our Chroma collection.
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
)

## This generate a persistant collection. If you want to clear all cached information, delete 'data' in the same directory.
# collection = Chroma("conversation_memory", embeddings, persist_directory=f'{IMG_ROOT_PATH}/chroma')

# This generate a temporary collection.
collection = Chroma("conversation_memory", embeddings)
os.makedirs(IMG_ROOT_PATH, exist_ok=True)


def generate_response(image: Image.Image, message: str, **kwargs):
    global model, processor
    prompt = f"USER: <image>\n{message}\nASSISTANT:"
    logger.info(f" ==== prompt ====\n{message}")
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    texts = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    responses = [
        {"content": text.split("ASSISTANT:")[-1].strip(), "index": i}
        for i, text in enumerate(texts)
    ]
    logger.info(f" ==== responses ====\n{responses}")
    return responses


def generate_output(conversation:list, user_input: str, img_obj):
    conversation.append([user_input, None])
    conversation.append([(img_obj.name,), None])
    yield conversation
    img_id = str(uuid3(NAMESPACE_DNS, Path(img_obj.name).name))
    img_save_pth = Path(f"data/{img_id}.jpeg")
    shutil.copy(img_obj.name, img_save_pth)
    # generate image description
    query = "Describe the image in detail."
    responses = generate_response(
        Image.open(img_save_pth),
        query,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
    )
    img_desc = responses[0]["content"]

    # find related image information from collection
    docs_with_score = collection.similarity_search_with_score(query=img_desc, k=1)
    logger.info(f" ==== find from collection ====\n{docs_with_score}")
    related_doc_info = []
    if len(docs_with_score) > 0:
        docs_with_score = sorted(docs_with_score, reverse=True, key=lambda x: x[1])
        for doc, score in docs_with_score:
            if score >= THRESHOLD:
                related_doc_info.append(doc)
    if related_doc_info:
        most_related_doc_info = related_doc_info[0]
    else:
        most_related_doc_info = None

    # Add image description to collection
    collection.add_texts(
        texts=[img_desc], metadatas=[{"path": str(img_save_pth.absolute())}]
    )
    logger.debug(f" ==== add texts to vector storage ====\n{img_desc}")

    # Visual question answer
    responses = generate_response(
        Image.open(img_save_pth),
        user_input,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
    )
    response = responses[0]["content"]

    # Yield model output for given query and image.
    conversation.append([None, response.strip()])
    yield conversation
    # Yield model output for related image
    if most_related_doc_info:
        conversation.append([None, "I found a similar image:"])
        conversation.append([None, (most_related_doc_info.metadata["path"],)])
        yield conversation


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=800)
        with gr.Column():
            with gr.Row():
                text_box = gr.Textbox(
                    lines=10,
                    scale=7,
                    placeholder="Enter text and upload an image, press the button to submit"
                )
                image_box = gr.File(scale=3, file_types=["image"])
            with gr.Row():
                submit_btn = gr.Button("submit")
                clear_btn = gr.ClearButton(
                    components=[chatbot, text_box, image_box]
                )
    submit_btn.click(
        fn=generate_output,
        inputs=[chatbot, text_box, image_box],
        outputs=[chatbot],
    )
demo.launch(server_port=7860)
