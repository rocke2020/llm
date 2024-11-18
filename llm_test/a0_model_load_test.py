import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

model_dir = '/mnt/nas1/models/MaziyarPanahi/Calme-7B-Instruct-v0.2'
# Use a pipeline as a high-level helper
# pipe = pipeline("text-generation", model=model_dir)

# Load model directly

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
logger.info('end')