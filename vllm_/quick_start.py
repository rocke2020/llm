import os
import sys
from vllm import LLM, SamplingParams
from icecream import ic
from loguru import logger


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


model_id = "/mnt/nas1/models/meta-llama/pretrained_weights/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    tensor_parallel_size=1,
)
tokenizer = llm.get_tokenizer()

conversations = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
)
ic(conversations)
outputs = llm.generate(
    [conversations],
    SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    ),
)
ic(outputs)
