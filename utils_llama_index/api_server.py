"""

"""

import os

from contextlib import asynccontextmanager
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
sys.path.append(os.path.abspath('.'))
from utils_llama_index.llm_loader import load_hf_llm
from llama_index.core.llms import ChatMessage

# Set up limit request time
# EventSourceResponse.DEFAULT_PING_INTERVAL = 1000
llm = load_hf_llm()


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


class InputRequest(BaseModel):
    input_str: str


class ChatInputRequest(BaseModel):
    messages: List[dict]


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/api/generate")
async def complete(request: InputRequest):
    """
    Returns a dict, example:
        {'text': "Hello! I'm doing well, thank you for asking. How about you? I'm here to help with any questions or topics you'd like to discuss.\n\nIf you're looking for some conversation starters, here are a few ideas:\n\n1. What are your interests or hobbies?\n2. Have you read any good books or watched any interesting movies or TV shows lately?\n3. What are your plans for the weekend or any upcoming events you're looking forward to?\n4. Do you have any favorite travel destinations or places you'd like to visit in the future?\n5. Are there any interesting news stories or current events you'd like to discuss?\n\nFeel free to share your thoughts on any of these topics or any other subject you'd like to explore!", 
        'additional_kwargs': {}, 
        'raw': {'model_output': {}}, 
        'logprobs': None, 
        'delta': None}

        dict_keys(['text', 'additional_kwargs', 'raw', 'logprobs', 'delta'])

    """
    response = llm.complete(request.input_str)
    return response


@app.post("/api/chat")
async def chat(request: ChatInputRequest):
    """
    Returns ChatResponse
    """
    messages = [ChatMessage(**msg) for msg in request.messages]
    response = llm.chat(messages)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)
