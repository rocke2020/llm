import transformers
import logging, sys, os
from transformers import AutoTokenizer
sys.path.append(os.path.abspath('.'))
import logging
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
from itertools import chain
import evaluate
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(lineno)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
cache_dir = '/mnt/nas1/huggingface/cache'
gpt2_model_name_or_path = '/mnt/nas1/models/gpt2'
local_eval_accuracy_file = '/home/qcdong/codes/evaluate/metrics/accuracy/accuracy.py'


def test():
    """  """
    metric = evaluate.load(local_eval_accuracy_file)
    # metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        ic(labels)
        preds = preds[:, :-1].reshape(-1)
        ic(preds)
        return metric.compute(predictions=preds, references=labels)
    
    eval_preds = (
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 0]]), 
    )
    r = compute_metrics(eval_preds)
    ic(r)


test()