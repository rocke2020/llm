import os
import socket
import sys

from icecream import ic
from loguru import logger
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath("."))

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


def get_local_ip(only_last_address=True) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("192.255.255.255", 1))
        local_ip = s.getsockname()[0]
    except OSError as e:
        logger.info("cannot get ip with error %s\nSo the local ip is 127.0.0.1", e)
        local_ip = "127.0.0.1"
    finally:
        s.close()
    logger.info("full local_ip %s, only_last_address %s", local_ip, only_last_address)
    if only_last_address:
        local_ip = local_ip.split(".")[-1]
    return local_ip


model_path = "/mnt/nas1/models/meta-llama/pretrained_weights/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
ip = get_local_ip()
if ip == "123":
    model_path = "/mnt/sde/models/Meta-Llama-3-8B-Instruct"

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

llm = LLM(
    model=model_path,
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
