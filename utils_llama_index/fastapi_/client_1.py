from pathlib import Path
import os, sys, shutil, json
from datetime import datetime
from collections import defaultdict
import re, random, math, logging
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic
from loguru import logger
import requests

sys.path.append(os.path.abspath('.'))


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

base_url = "http://127.0.0.1:8888/translate/"


def translate_text(input_str):
    data = {"input_str": input_str}
    response = requests.post(base_url, json=data)
    response.raise_for_status()
    response_json = response.json()
    translated_text = response_json["translated_text"]
    return translated_text


response = translate_text('test input string')
print(f'{response = }')
