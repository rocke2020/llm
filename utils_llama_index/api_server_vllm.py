"""

"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.abspath("."))
from vllm import LLM, SamplingParams

from utils_llama_index.model_comm import get_model_path


def load_vllm(
    model_name="Llama-3",
):
    model_path = get_model_path(model_name)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    return llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


llm = load_vllm()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VllmInputRequest(BaseModel):
    prompt_token_ids: List[List[int]]
    max_new_tokens: int = 1024
    temperature: float = 0.00


@app.post("/api/vllm_generate")
async def vllm_generate(request: VllmInputRequest):
    """
    Returns a list, example:
        orig response:
        [RequestOutput(request_id=0, prompt=None, prompt_token_ids=[128000, 128006, 9125, 128007, 271, 2675, 527, 264, 55066, 6369, 6465, 889, 2744, 31680, 304, 55066, 6604, 0, 128009, 128006, 882, 128007, 271, 15546, 527, 499, 30, 128009, 128006, 78191, 128007, 271], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text="Arrr, me hearty! Me name be Captain Chatbot, the scurviest pirate to ever sail the Seven Seas! Me be a swashbucklin' chatbot, here to regale ye with tales o' adventure, answer yer questions, and maybe even share a few sea shanties to get ye in the mood fer a swashbucklin' good time! So hoist the colors, me matey, and let's set sail fer a chat like no other!", token_ids=[9014, 81, 11, 757, 82651, 0, 2206, 836, 387, 22022, 13149, 6465, 11, 279, 1156, 324, 10176, 478, 55066, 311, 3596, 30503, 279, 31048, 93496, 0, 2206, 387, 264, 2064, 1003, 65, 1983, 3817, 6, 6369, 6465, 11, 1618, 311, 1239, 1604, 20043, 449, 37565, 297, 6, 18427, 11, 4320, 55295, 4860, 11, 323, 7344, 1524, 4430, 264, 2478, 9581, 559, 519, 552, 311, 636, 20043, 304, 279, 20247, 18728, 264, 2064, 1003, 65, 1983, 3817, 6, 1695, 892, 0, 2100, 11640, 380, 279, 8146, 11, 757, 30276, 88, 11, 323, 1095, 596, 743, 30503, 18728, 264, 6369, 1093, 912, 1023, 0, 128009], cumulative_logprob=-11.95750205218792, logprobs=None, finish_reason=stop, stop_reason=128009)], finished=True, metrics=RequestMetrics(arrival_time=1714194067.9569736, last_token_time=1714194067.9569736, first_scheduled_time=1714194067.95995, first_token_time=1714194068.014371, time_in_queue=0.0029764175415039062, finished_time=1714194070.7369564), lora_request=None)]
        
        simple response:
        [[{'index': 0, 'text': 'Here is the list of peptide names and their corresponding sequences:\n\n* Copper tripeptides: Cu-GHK\n* tetrapeptide: PKEK\n* manganese tripeptide-1: unavailable\n* silk fibroin peptide: unavailable\n* tat: RKKRRQRRR\n* CPPAIF: unavailable', 'token_ids': [8586, 374, 279, 1160, 315, 72249, 5144, 323, 872, 12435, 24630, 1473, 9, 43640, 2463, 375, 74489, 25, 27560, 12279, 58547, 198, 9, 28953, 20432, 47309, 25, 393, 3472, 42, 198, 9, 87934, 2463, 375, 47309, 12, 16, 25, 36087, 198, 9, 41044, 16178, 299, 258, 72249, 25, 36087, 198, 9, 72813, 25, 432, 51557, 8268, 48, 8268, 49, 198, 9, 38771, 32, 2843, 25, 36087, 128009], 'cumulative_logprob': -5.634563509015031, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': 128009, 'lora_request': None}]]

    """
    sampling_params = SamplingParams(
        stop_token_ids=[128001, 128009],
        temperature=request.temperature,
        max_tokens=request.max_new_tokens,
    )
    response = llm.generate(
        prompt_token_ids=request.prompt_token_ids, sampling_params=sampling_params
    )
    simple_response = []
    for item in response:
        simple_response.append(item.outputs)
    return simple_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
