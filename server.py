"""
This script implements an API for the TinyLLaVA model,
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
It's designed to be run as a web server using FastAPI and uvicorn,
making the TinyLLaVA model accessible through OpenAI Client.
"""

import time
import torch
import uvicorn

from io import BytesIO
from urllib.request import urlopen
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union, cast, Dict, Any
from loguru import logger
from pydantic import BaseModel, Field

# from sentence_transformers import SentenceTransformer

from sse_starlette.sse import EventSourceResponse
from loguru import logger
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from pathlib import Path
from uuid import uuid1

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageUrl(BaseModel):
    url:str
    detail:Optional[str]='auto'

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class UserMessage(BaseModel):
    content: Union[str, List[TextContentPart | ImageContentPart]]
    role: Literal["user"]
    name: Optional[str] = None


class AssistantMessage(BaseModel):
    content: Optional[str] = None
    role: Literal["assistant"]
    name: Optional[str] = None


class SystemMessage(BaseModel):
    content: str
    role: Literal["system"]
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    name: Optional[str] = None


# for ChatCompletionRequest


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[SystemMessage | UserMessage | AssistantMessage]
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: SystemMessage | UserMessage | AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="tiny-llava-v1-hf")
    return ModelList(data=[model_card])


def build_prompt(messages: list[Dict[str,Any]]):
    prompt = ""
    image = None
    for m in messages:
        if m['role'] == "user":
            prompt += "USER: "
            if isinstance(m['content'], str):
                prompt += m["content"]
            else:
                for part in m['content']:
                    if isinstance(part, str):
                        prompt += part
                    else:
                        prompt += "<image>\n"
                        image = part
        elif m['role'] == "assistant":
            prompt += f"ASSISTANT: {m['content']}"
        prompt+='ASSISTANT: '
    return prompt, image


def preprocess_messages(
    raw_messages: List[UserMessage | SystemMessage | AssistantMessage],
):
    """
    ignore system message
    """
    messages = []
    num_of_image = 0
    for m in raw_messages:
        content = None
        match m.role:
            case "user":
                if isinstance(m.content, str):
                    content = m.content
                else:
                    content = []
                    for part in m.content:
                        match part.type:
                            case "text":
                                content.append(part.text)
                            case "image_url":
                                if num_of_image == 1:
                                    raise HTTPException(
                                        400, detail="More than 1 image provided"
                                    )
                                num_of_image += 1
                                with urlopen(part.image_url.url) as response:
                                    data = response.read()
                                    content.append(Image.open(BytesIO(data)))
                messages.append({"role": m.role, "content": content})
            case "assistant":
                if m.content is None:
                    continue
                content = m.content
                messages.append({"role": m.role, "content": content})
    return messages


def generate_response(model, processor, raw_messages: list, **kwargs):
    messages = preprocess_messages(raw_messages)
    prompt, image = build_prompt(messages)
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")
    logger.debug(f" ==== prompt ====\n{prompt}")
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    raw_responses = model.generate(**inputs, **kwargs)
    return postprocess_responses(raw_responses)


def postprocess_responses(raw_responses: list[Dict[str, Any]]):
    responses = [
        processor.decode(output, skip_special_tokens=True) for output in raw_responses
    ]
    logger.debug(f" ==== raw responses ====\n{responses}")
    responses = [response.split("ASSISTANT:")[-1].strip() for response in responses]
    responses = [response.replace('\n\n', '\n') for response in responses]
    logger.debug(f" ==== responses ====\n{responses}")
    return responses


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # TODO: request.model check
    global model, processor

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    if request.stream:
        raise HTTPException(status_code=400, detail="Not support stream")
    if request.tools:
        raise HTTPException(status_code=400, detail="Not support tools")
    kwargs = dict(
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens or 200,
        do_sample=True if request.top_p else False
    )
    logger.debug(f"==== generate kwargs ====\n{kwargs}")
    responses = generate_response(model, processor, request.messages, **kwargs)
    choices = []
    for i, response in enumerate(responses):
        message = AssistantMessage(role="assistant", content=response)
        choice_data = ChatCompletionResponseChoice(
            index=i,
            message=message,
            finish_reason="stop",  # All finish reasons are stop
        )
        choices.append(choice_data)
    usage = UsageInfo()
    return ChatCompletionResponse(
        model=request.model,
        id=str(uuid1()),
        choices=[choice_data],
        object="chat.completion",
        usage=usage,
    )


if __name__ == "__main__":
    model_id = "bczhou/tiny-llava-v1-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()  # type:ignore
    processor = AutoProcessor.from_pretrained(model_id)
    uvicorn.run(app, host="0.0.0.0", port=6006, workers=1)
