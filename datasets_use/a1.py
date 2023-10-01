import transformers
import logging, sys, os
from datasets import load_dataset
from transformers import AutoTokenizer
sys.path.append(os.path.abspath('.'))
import logging
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
from itertools import chain


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(lineno)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
cache_dir = '/mnt/nas1/huggingface/cache'
gpt2_model_name_or_path = '/mnt/nas1/models/gpt2'
tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name_or_path)
ic(tokenizer.model_max_length)


def load_wikitext_2_raw_v1(offline=True, verbose=True):
    """
    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 36718
        })
        test: Dataset({
            features: ['text'],
            num_rows: 4358
        })
        validation: Dataset({
            features: ['text'],
            num_rows: 3760
        })
    })
    """
    wikitext_2_raw_v1_dir = '/mnt/nas1/huggingface/wikitext/wikitext-2-raw-v1'
    logger.info('load_wikitext_2_raw_v1')
    if offline:
        data_files = {
            'train': wikitext_2_raw_v1_dir + '/train/' + '0000.parquet',
            'test': wikitext_2_raw_v1_dir + '/test/' + '0000.parquet',
            'validation': wikitext_2_raw_v1_dir + '/validation/' + '0000.parquet',
        }
        raw_datasets = load_dataset(
            'parquet',
            data_files=data_files,
            cache_dir=cache_dir,
        )
    else:
        raw_datasets = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            cache_dir=cache_dir,
        )
    logger.info(raw_datasets)
    if verbose:
        train_dataset = raw_datasets['train']
        count = 0
        for item in train_dataset:
            logger.info(item)
            count += 1
            if count > 10:
                break
    return raw_datasets


def test():
    """  """
    from transformers.testing_utils import CaptureLogger

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    text_column_name = 'text'
    raw_datasets = load_wikitext_2_raw_v1(verbose=False)

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    train_dataset = raw_datasets['train']
    logger.info(train_dataset[10])
    for i in range(4):
        logger.info(len(raw_datasets['train'][i]['text']))
        logger.info(raw_datasets['train'][i]['text'])
    output = tokenize_function(train_dataset[:4])
    logger.info(output)

    column_names = list(raw_datasets["train"].features)
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    logger.info(tokenized_datasets)
    for i in range(4):
        logger.info(len(tokenized_datasets['train'][i]['input_ids']))
        logger.info(tokenized_datasets['train'][i]['input_ids'])
    assert (tokenized_datasets['train'][:4] == output)

    block_size = 1024

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        # len(examples), list(examples.keys()), len(examples['input_ids'])
        # 2, ['input_ids', 'attention_mask'], 1000 which is batch_size.
        logger.info('%s, %s, %s', len(examples), list(examples.keys()), len(examples['input_ids']))
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        # load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    train_dataset = lm_datasets["train"]
    logger.info(lm_datasets)
    logger.info(len(train_dataset))


test()