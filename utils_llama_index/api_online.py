import os
import sys

from loguru import logger
from openai import OpenAI

sys.path.append(os.path.abspath("."))
from utils_llama_index.api_client import system_bio

client = OpenAI(base_url="https://api.perplexity.ai")


def llama3_70b_generate(input_txt, temperature=0.0):
    """ """
    return call_perplexity(input_txt, "llama-3-70b-instruct", temperature)


def llama3_sonar(input_txt, temperature=0.0):
    """ """
    return call_perplexity(input_txt, "llama-3-sonar-large-32k-chat", temperature)


def call_perplexity(input_txt, model_type, temperature):
    messages = [system_bio, {"role": "user", "content": input_txt}]
    response = client.chat.completions.create(
        model=model_type,
        messages=messages,
        temperature=temperature,
    )
    # print(response)
    return response.choices[0].message.content


def mixtral_8_7b(input_txt, temperature=0.0):
    return call_perplexity(input_txt, "mixtral-8x7b-instruct", temperature)


if __name__ == "__main__":
    logger.info("end")