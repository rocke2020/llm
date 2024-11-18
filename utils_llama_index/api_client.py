import os
import sys
from functools import cache

import requests
from loguru import logger
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("."))
from utils_llama_index.model_comm import get_model_path

BASE_URL = "http://127.0.0.1:8001/"
SYSTEM_PROMPT_LLAMA2 = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
"""
SYSTEM_PROMPT_BIO = """
You are a biologist that answers questions, based on the given contents from a scientific article. Here are some rules you always follow:
- answer the question only using the provided context and do not make up the answer with prior knowledge.
- avoid creating output with gibberish text.
"""
SYSTEM_PROMPT_BIO_PEPTIDE = """
You are a biologist, and your task is to extract and list peptide sequences from provided scientific article contents.
"""
system_input_general = {"role": "system", "content": SYSTEM_PROMPT_LLAMA2}
system_bio = {"role": "system", "content": SYSTEM_PROMPT_BIO_PEPTIDE}
user_input_base = {"role": "user", "content": "Who are you?"}


context_and_question = (
    "Context information is below.\n---------------------\n"
    "{context}\n--------------------\n"
    "Given the context information and not prior knowledge, answer the query:\n"
    "{question}"
)


def vllm_generate_with_tokenizer(
    input_txt, temperature=0.0, system_input=system_bio, base_url=BASE_URL
):
    tokenizer = get_tokenizer()
    reply_text = vllm_generate(
        tokenizer, input_txt, temperature, system_input, base_url
    )
    return reply_text


@cache
def get_tokenizer(model_name="Llama-3"):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def create_user_input_dict(input_txt):
    return {"role": "user", "content": input_txt}


def vllm_generate(
    tokenizer, input_txt, temperature=0.0, system_input=system_bio, base_url=BASE_URL
):
    url = base_url + "api/vllm_generate"
    messages = [system_input, create_user_input_dict(input_txt)]
    input_ids = [tokenizer.apply_chat_template(messages, tokenize=True)]
    data = {"prompt_token_ids": input_ids, 'temperature': temperature}
    response = requests.post(url, json=data)
    response.raise_for_status()
    response_json = response.json()
    reply_text = response_json[0][0]['text']
    return reply_text


def chat(input_txt, only_return_text=True, debug=False, base_url=BASE_URL):
    url = base_url + "api/chat"
    user_input = create_user_input_dict(input_txt)
    data = {"messages": [system_bio, user_input]}
    response = requests.post(url, json=data)
    response.raise_for_status()
    response_json = response.json()
    if debug:
        logger.info(f'{response_json = }')
    if only_return_text:
        return response_json["message"]["content"]
    return response_json


def chat_with_context(
    question, context, only_return_text=True, debug=False, base_url=BASE_URL
):
    url = base_url + "api/chat"
    content = context_and_question.format(context=context, question=question)
    user_input = create_user_input_dict(content)
    data = {"messages": [system_bio, user_input]}
    response = requests.post(url, json=data)
    response.raise_for_status()
    response_json = response.json()
    if debug:
        logger.info(f'{response_json = }')
    if only_return_text:
        return response_json["message"]["content"]
    return response_json


def generate_api(input_txt, only_return_text=True, base_url=BASE_URL):
    """ Generate completion from input text using the API.
    
    dict_keys(['text', 'additional_kwargs', 'raw', 'logprobs', 'delta'])
    """
    url = base_url + "api/generate"
    data = {"input_str": input_txt}
    response = requests.post(url, json=data)
    response.raise_for_status()
    response_json = response.json()
    # logger.info(f'{response_json = }')
    if only_return_text:
        return response_json["text"]
    return response_json


def check_health(base_url=BASE_URL):
    url = base_url + "health"
    response = requests.get(url)
    response.raise_for_status()
    print(response)


def query_with_context(question, context, only_return_text=True):
    input_text = context_and_question.format(context=context, question=question)
    response = generate_api(input_text, only_return_text)
    return response


def query_with_nodes(question, response_nodes, only_return_text=True):
    context = "\n".join(
        [
            "Context {}:\n".format(index) + node.text
            for index, node in enumerate(response_nodes)
        ]
    )
    response = query_with_context(question, context, only_return_text)
    return response


def query_with_retriever(question, context_retriever, only_return_text=True):
    response_nodes = context_retriever.retrieve(question)
    response = query_with_nodes(question, response_nodes, only_return_text)
    return response


if __name__ == "__main__":
    logger.info(f"{BASE_URL = }")
    # check_health()
    input_str = "Hello, how are you today?"
    # query_response = generate_api(input_str)
    # query_response = chat(input_str)
    # print(query_response)