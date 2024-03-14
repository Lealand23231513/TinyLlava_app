"""
This script implements an API for the TinyLLaVA model,
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
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
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextIteratorStreamer
from threading import Thread
from PIL import Image
from pathlib import Path
from uuid import uuid1
from sse_starlette.sse import EventSourceResponse 

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
    role: Literal["user", "assistant", "system"]
    content: Optional[str] = None


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
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    


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


def preprocess_messages(
    raw_messages: List[UserMessage | SystemMessage | AssistantMessage],
):
    """
    ignore system message
    """
    prompt = ""
    image = None
    num_of_image = 0
    for m in raw_messages:
        match m.role:
            case "user":
                prompt += "USER: "
                if isinstance(m.content, str):
                    prompt+= m.content
                else:
                    for part in m.content:
                        match part.type:
                            case "text":
                                prompt+=part.text
                            case "image_url":
                                if num_of_image == 1:
                                    raise HTTPException(
                                        400, detail="More than 1 image provided"
                                    )
                                num_of_image += 1
                                with urlopen(part.image_url.url) as response:
                                    data = response.read()
                                    image=Image.open(BytesIO(data))
                                prompt+="<image>\n"
            case "assistant":
                if m.content is None:
                    continue
                prompt += f"ASSISTANT: {m.content}"
    prompt+='ASSISTANT: '
    return prompt, image


def generate_response(model, processor, raw_messages: list, stream=False, **generation_kwargs):
    prompt, image = preprocess_messages(raw_messages)
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")
    logger.debug(f" ==== prompt ====\n{prompt}")
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    if stream:
        streamer = TextIteratorStreamer(processor,skip_special_tokens=True)
        generation_kwargs.update({"streamer":streamer})
        generation_kwargs.update(inputs)
        thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        return postprocess_stream_responses(streamer)
    
    raw_responses = model.generate(**inputs, **generation_kwargs)
    return postprocess_responses(raw_responses)

def postprocess_stream_responses(generator):
    start_flag=False
    for chunk_text in generator:
        chunk_text = cast(str, chunk_text)
        if 'ASSISTANT:' in chunk_text:
            continue
        if start_flag==False and chunk_text.isspace():
            continue
        if start_flag==False:
            start_flag=True
        chunk_text.replace('\n\n', '\n')
        delta = DeltaMessage(
            content=chunk_text,
            role="assistant",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id=str(uuid1()),
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(
            role="assistant"
        ),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id=str(uuid1()),
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'

def postprocess_responses(raw_responses: list[Dict[str, Any]]):
    responses = [
        processor.decode(output, skip_special_tokens=True) for output in raw_responses
    ]
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
    if request.tools:
        raise HTTPException(status_code=400, detail="Not support tools")
    generation_kwargs = dict(
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens or 200,
        do_sample=True if request.top_p else False
    )
    logger.debug(f"==== generate kwargs ====\n{generation_kwargs}\nstream={request.stream}")
    if request.stream:
        responses_generator = generate_response(model, processor, request.messages, request.stream, **generation_kwargs)
        return EventSourceResponse(responses_generator, media_type="text/event-stream")
    responses = generate_response(model, processor, request.messages, **generation_kwargs)
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
