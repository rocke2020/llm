import torch
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.huggingface.base import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
INST_BEGIN = "[INST] "
INST_END = " [/INST]"
model_dir = '/mnt/nas1/models/MaziyarPanahi/Calme-7B-Instruct-v0.2'

"""  Mistral-7B-Instruct
Instruction format
In order to leverage instruction fine-tuning, your prompt should be surrounded by [INST] and [/INST] tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.

text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"[INST] Do you have mayonnaise recipes? [/INST]"

"""


def completion_to_prompt(completion):
    return f"<s>[INST] {completion} [/INST]"


def messages_to_prompt(messages):
    prompt = BOS_TOKEN
    for message in messages:
        if message.role == "user":
            prompt += f"{INST_BEGIN}{message.content}{INST_END}"
        elif message.role == "assistant":
            prompt += f"{message.content}{EOS_TOKEN}"

    return prompt


def load_llama_cpp_model(
    model_path="/mnt/nas1/models/MaziyarPanahi/Calme-7B-Instruct-v0.5-GGUF/Calme-7B-Instruct-v0.5.Q4_K_M.gguf",
):
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=2048,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=30000,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


def load_hf_llm(
    model_path="/mnt/nas1/models/MaziyarPanahi/Calme-7B-Instruct-v0.2",
):
    use_8bit = 1
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
    logger.info(f"{use_8bit = }")
    llm = HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        context_window=4096,
        max_new_tokens=2048,
        # generate_kwargs={"temperature": 0.01, "do_sample": True, 'pad_token_id': 2},
        generate_kwargs={"do_sample": False, "pad_token_id": 2},
        # generate_kwargs={"temperature": 0.01, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="auto",
        model_kwargs={
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
        },
    )
    # print(llm.get_memory_footprint())
    return llm



if __name__ == "__main__":
    logger.info("starts")
    # llm = load_llama_cpp_model()

    llm = load_hf_llm()
    # logger.info(f'{llm._tokenizer = }')
    response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
    logger.info("\n" + response.text.strip())
    logger.info("end")
