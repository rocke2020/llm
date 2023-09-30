import os; import psutil; import timeit
from datasets import load_dataset
import logging


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')

logger.info('starts')
cache_dir = '/mnt/nas1/huggingface/cache'
