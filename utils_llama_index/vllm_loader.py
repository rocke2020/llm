import os
import sys

from loguru import logger
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath("."))
from utils_llama_index.model_comm import get_model_path


def load_vllm(model_name="Llama-3"):
    model_path = get_model_path(model_name)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    return llm


def test():
    llm = load_vllm()
    tokenizer = llm.get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]    
    conversations = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
    )
    logger.info(conversations)
    outputs = llm.generate(
        prompt_token_ids=[conversations],
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        ),
    )
    logger.info(outputs)

if __name__ == "__main__":
    test()