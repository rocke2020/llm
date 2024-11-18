import torch
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.huggingface.base import HuggingFaceLLM
from loguru import logger
from transformers import BitsAndBytesConfig

model_paths = [
    "/mnt/nas1/models/openchat-3.5-0106",
    "/mnt/nas1/models/Nexusflow/Starling-LM-7B-beta",
    "/mnt/nas1/models/mistralai/Mistral-7B-Instruct-v0.2",
    "/mnt/nas1/models/llama/pretrained_weights/llama2-7b-chat-hf",
]

SYSTEM_PROMPT_LLAMA2 = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""


def get_query_wrapper_prompt(model_path):
    if "llama2-7b-chat-hf" in model_path:
        query_wrapper_prompt = PromptTemplate(
            "[INST]<<SYS>>\n" + SYSTEM_PROMPT_LLAMA2 + "<</SYS>>\n\n{query_str}[/INST] "
        )
    elif 'openchat-3.5' in model_path or "Starling-LM-7B-beta" in model_path:
        query_wrapper_prompt = PromptTemplate(
            "GPT4 Correct User: {query_str}<|end_of_turn|>GPT4 Correct Assistant:"
        )
    elif "Mistral-7B-Instruct-v0.2" in model_path:
        query_wrapper_prompt = PromptTemplate("<s>[INST] {query_str} [/INST]")
    return query_wrapper_prompt


def load_hf_llm(
    model_path=model_paths[0],
    use_8bit=1,
    max_length=8192,
    max_new_tokens=2048,
    do_sample=False,
    temperature=0.01,
):
    """Load a HuggingFace LLM model."""
    logger.info(f'{model_path = }')
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    if use_8bit:
        quantization_config = quantization_config_8bit
    else:
        quantization_config = quantization_config_4bit
    context_window = max_length - max_new_tokens
    logger.info(f"{use_8bit =} {max_length = } {context_window = } {max_new_tokens = }")

    query_wrapper_prompt = get_query_wrapper_prompt(model_path)
    if do_sample:
        generate_kwargs = {"temperature": temperature, "do_sample": True}
    else:
        generate_kwargs = {"do_sample": False}
    llm = HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        # generate_kwargs={"temperature": 0.01, "do_sample": True, 'pad_token_id': 2},
        generate_kwargs=generate_kwargs,
        # generate_kwargs={"temperature": 0.01, "top_k": 50, "top_p": 0.95},
        query_wrapper_prompt=query_wrapper_prompt,
        device_map="auto",
        model_kwargs={
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
        },
    )
    return llm


## Old code styple:
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# embed_model = HuggingFaceEmbedding(
#     model_name="/home/cymei/bge-small-en",device='cuda:3'
# )


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["LLAMA_INDEX_CACHE_DIR"] = "/mnt/nas1/models/llama_index_cache"
    Settings.embed_model = "local:/mnt/nas1/models/BAAI/bge-small-en-v1.5"
    _llm = load_hf_llm()

    input_text = "Hello, how are you?"
    response = _llm.complete(input_text)
    print(response)
