# IMPORTANT:
# Before running this demo, you should fill your openai api-key in '.env. If you can't find a file named '.env', make a copy
# of '.env.template' and rename it to '.env'
import os
import numpy as np
from loguru import logger
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from PIL import Image
from uuid import uuid1
from pathlib import Path
from typing import Tuple, Optional

MAX_NEW_TOKENS = 200
IMG_ROOT_PATH = "data/"
THRESHOLD = 0.01


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


def generate_output(
    user_input: str, img: np.ndarray
) -> Tuple[str, str, Optional[np.ndarray]]:
    global model, processor, collection
    img_id = str(uuid1())
    img_save_pth = Path(f"data/{img_id}.jpeg")
    img_obj = Image.fromarray(img)
    img_obj.save(img_save_pth)
    # generate image description
    query = "Describe the image in detail."
    responses = generate_response(
        img_obj, query, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7
    )
    img_desc = responses[0]["content"]

    # find related image information from collection
    docs_with_score = collection.similarity_search_with_score(query=img_desc, k=1)
    logger.info(f" ==== find from collection ====\n{docs_with_score}")
    related_img_info_lst=[]
    related_img_objs=[]
    if len(docs_with_score) > 0:
        docs_with_score = sorted(
            docs_with_score,
            reverse=True,
            key = lambda x:x[1]
        )
        for doc, score in docs_with_score:
            if score>=THRESHOLD:
                related_img_info_lst.append(
                    f"**description:**\n{doc.page_content}"
                )
                related_img_objs.append(
                    Image.open(doc.metadata["path"])
                )
    if related_img_info_lst:
        related_img_info = related_img_info_lst[0]
        related_img_obj = related_img_objs[0]
    else:
        related_img_info = "No related image found."
        related_img_obj = None
    
    # add image description to collection
    collection.add_texts(
        texts=[img_desc], metadatas=[{"path": str(img_save_pth.absolute())}]
    )
    logger.debug(f" ==== add texts to vector storage ====\n{img_desc}")

    # Visual question answer
    responses = generate_response(
        img_obj,
        user_input,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
    )
    response = responses[0]["content"]

    # Return model output.
    return (
        f"# Response\n{response.strip()}",
        f"# Related Image Infomation\n{related_img_info.strip()}",
        np.array(related_img_obj) if related_img_obj else None,
    )



with gr.Interface(
    generate_output,
    inputs=[
        gr.Textbox(
            label="Input text",
            scale=7,
        ),
        gr.Image(label="Input image"),
    ],
    outputs=[
        gr.Markdown(value="# Response", line_breaks=True),
        gr.Markdown(value="# Related Image Infomation", line_breaks=True),
        gr.Image(label="related image"),
    ],
    clear_btn=gr.Button("Clear"),
    title="TinyLLaVA language-Image QA demo",
    description="""
        **IMPORTANT:** You need to provide both an image and a text-based query.
        """,
    allow_flagging="never",
) as demo:
    pass
if __name__ == "__main__":
    load_dotenv()

    model_id = "bczhou/tiny-llava-v1-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, low_cpu_mem_usage=True, device_map="cuda"
    ).eval()# type:ignore
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
    demo.queue().launch(server_port=7860)
