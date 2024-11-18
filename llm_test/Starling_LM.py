import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers
from icecream import ic
from loguru import logger
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import BitsAndBytesConfig


model_dir = "/mnt/nas1/models/Nexusflow/Starling-LM-7B-beta"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
tokens = tokenizer(
    "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:"
).input_ids

# Single-turn
tokens = tokenizer(
    "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:"
).input_ids
assert tokens == [
    1,
    420,
    6316,
    28781,
    3198,
    3123,
    1247,
    28747,
    22557,
    32000,
    420,
    6316,
    28781,
    3198,
    3123,
    21631,
    28747,
]

# Multi-turn
tokens = tokenizer(
    "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:"
).input_ids
assert tokens == [
    1,
    420,
    6316,
    28781,
    3198,
    3123,
    1247,
    28747,
    22557,
    32000,
    420,
    6316,
    28781,
    3198,
    3123,
    21631,
    28747,
    15359,
    32000,
    420,
    6316,
    28781,
    3198,
    3123,
    1247,
    28747,
    1602,
    460,
    368,
    3154,
    28804,
    32000,
    420,
    6316,
    28781,
    3198,
    3123,
    21631,
    28747,
]

# Coding Mode
tokens = tokenizer(
    "Code User: Implement quicksort using C++<|end_of_turn|>Code Assistant:"
).input_ids
assert tokens == [
    1,
    7596,
    1247,
    28747,
    26256,
    2936,
    7653,
    1413,
    334,
    1680,
    32000,
    7596,
    21631,
    28747,
]


def get_device(device_id=0):
    """if device_id < 0: device = "cpu" """
    gpu_count = 0
    if device_id < 0:
        device = "cpu"
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > device_id:
            device = f"cuda:{device_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f"gpu_count {gpu_count}, device: {device}")
    return device


device = get_device()
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)
model_kwargs = {
    "quantization_config": quantization_config_8bit,
    "torch_dtype": torch.float16,
}
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_dir, device_map=device, **model_kwargs
)


def generate_response(prompt, remove_input=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if not remove_input:
        response_ids = outputs[0]
    else:
        response_ids = outputs[0][input_ids.size(1) :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text


# Single-turn conversation
prompt = "Hello, how are you?"
single_turn_prompt = (
    f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
)
response_text = generate_response(single_turn_prompt)
print("Response:", response_text)
response_text = generate_response(single_turn_prompt, remove_input=True)
print("\n\nremove_input Response:", response_text)

## Multi-turn conversation
prompt = "Hello"
follow_up_question = "How are you today?"
response = "Hi"
multi_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:"
response_text = generate_response(multi_turn_prompt)
print("\n\nMulti-turn conversation response:", response_text)

### Coding conversation
prompt = "Implement quicksort using C++"
coding_prompt = f"Code User: {prompt}<|end_of_turn|>Code Assistant:"
response = generate_response(coding_prompt)
print("\n\nCoding conversation response:", response)

"""Outputs 

Response: GPT4 Correct User: Hello, how are you? GPT4 Correct Assistant: Hello! I'm doing well, thank you for asking. As an AI language model, I'm always ready to assist and engage in conversations. How about you? How is your day going?

If you have any specific questions or topics you'd like to discuss, feel free to ask, and I'll do my best to help you.


remove_input Response: Hello! I'm doing well, thank you for asking. As an AI language model, I'm always ready to assist and engage in conversations. How about you? How is your day going?

If you have any specific questions or topics you'd like to discuss, feel free to ask, and I'll do my best to help you.


Multi-turn conversation response: GPT4 Correct User: Hello GPT4 Correct Assistant: Hi GPT4 Correct User: How are you today? GPT4 Correct Assistant: I'm functioning well, thank you for asking! As an AI language model, I'm always ready to assist and engage in conversations. How about you? How is your day going?


Coding conversation response: Code User: Implement quicksort using C++ Code Assistant: Here is an example of quicksort implemented in C++:

```cpp
#include <iostream>
#include <vector>

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        int pi = i + 1;
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();
    quick
"""