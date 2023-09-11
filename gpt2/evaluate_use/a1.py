import evaluate
from pathlib import Path
import os, sys, logging, shutil
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import re, random, math
import pandas as pd
import numpy as np
from pandas import DataFrame
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s',
    datefmt='%y-%m-%d %H:%M')
from datetime import datetime
from collections import defaultdict


SEED = 0
logger.info('seed %s', SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except Exception as identifier:
    logger.exception(identifier)

accuracy = evaluate.load("accuracy")