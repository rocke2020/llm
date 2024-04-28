"""  
https://docs.vllm.ai/en/latest/quantization/fp8_e4m3_kvcache.html

https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md
This utility extracts the KV cache scaling factors from a quantized HF (Hugging Face) model. The extracted scaling factors are saved to a JSON file, which can later be used by vLLM (variable-length language model) during runtime. This tool is particularly useful when the KV cache data type is FP8 and is intended for use on ROCm (AMD GPU) platforms.
"""
import os
import socket
import sys

from icecream import ic
from loguru import logger
from transformers import AutoTokenizer
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
tokenizer = AutoTokenizer.from_pretrained(model_path)
ip = get_local_ip()
if ip == "123":
    model_path = "/mnt/sde/models/Meta-Llama-3-8B-Instruct"
ic(model_path)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

"""  
tensor_parallel_size: 2, cause error to init the LLM.
[33m(raylet)[0m [2024-04-28 07:02:15,029 E 2264854 2264884] (raylet) file_system_monitor.cc:111: /tmp/ray/session_2024-04-28_07-02-03_007827_2264683 is over 95% full, available space: 23473643520; capacity: 982820896768. Object creation will fail if spilling is required.
[36m(RayWorkerWrapper pid=2268426)[0m INFO 04-28 07:02:13 utils.py:129] reading GPU P2P access cache from /home/qcdong/.config/vllm/gpu_p2p_access_cache_for_0,1.json
[36m(RayWorkerWrapper pid=2268426)[0m WARNING 04-28 07:02:13 custom_all_reduce.py:74] Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.
...
Error executing method initialize_cache. This might cause deadlock in distributed execution.
...
RuntimeError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN for details)
"""
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    # kv_cache_dtype="fp8",
    # quantization_param_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json"    
)
# tokenizer = llm.get_tokenizer()

conversations = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
)
ic(conversations)

outputs = llm.generate(
    prompt_token_ids=[conversations],
    sampling_params = SamplingParams(
        temperature=0.0,
        # top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[128001, 128009],
    ),
)
ic(outputs)