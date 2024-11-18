""" finetune the base model, not chat model """
import os
os.environ.get('CUDA_VISIBLE_DEVICES')

# load input text sentences
from pandas import read_json, Series
CTD_RE_V1 = read_json('../label_studio/export/CTD_RE_v1.json').set_index('id')
sentences = Series(data = [row['text'] for row in CTD_RE_V1.data], index=CTD_RE_V1.index)


# combine input sentences and annotations to a jsonl file
from pathlib import Path
from re import findall
from json import load, dump

pathlist = Path('../data/random_2000/output_claude/langchain_annotated').glob('*.json')
instances = []
output_file = 'claude_annotations.jsonl'
null_count = 0
with open(output_file, 'w') as outfile:
    for path in pathlist:
        task_id = int(findall(r'\d+', str(path))[1])
        with open(path) as json_file:
            output = load(json_file)
            if output and 'relations' in output:
                if isinstance(output['relations'], list):
                    json_obj = {'input_sentence': sentences[task_id], 'relations': output['relations']}
                else:
                    null_count += 1
                    json_obj = {'input_sentence': sentences[task_id], 'relations': []}
            else:
                null_count += 1
                json_obj = {'input_sentence': sentences[task_id], 'relations': []}
            dump(json_obj, outfile)
            outfile.write('\n')


# load train and validation set
from datasets import load_dataset
ds = load_dataset('json', data_files='claude_annotations.jsonl', split="train")
ds = ds.train_test_split(test_size=0.1)

train_dataset = ds['train']
eval_dataset = ds['test']
print(train_dataset)
print(eval_dataset)
"""  
Dataset({
    features: ['input_sentence', 'relations'],
    num_rows: 1800
})
Dataset({
    features: ['input_sentence', 'relations'],
    num_rows: 200
})
"""
print(train_dataset[2])
"""  
{'input_sentence': 'The results show that exposure to arsenite, increased miR-15b levels and induced M2 polarization of THP-1 cells.',
 'relations': [{'subject_entity': {'entity_name': 'arsenite',
    'entity_type': 'Chemical'},
   'relation_phrase': 'increases',
   'object_entity': {'entity_name': 'miR-15b',
    'entity_type': 'Gene/Protein'}}]}
"""

def format_relation(relations):
    relation_str = ""
    if relations != None:
        for relation in relations:
            if relation != None:
                relation_str += "(("
                if 'subject_entity' in relation and relation['subject_entity'] != None:
                    if 'entity_name' in relation['subject_entity'] and relation['subject_entity']['entity_name'] != None:
                        relation_str += relation['subject_entity']['entity_name'] + ", "
                    else:
                        relation_str += ","
                    if 'entity_type' in relation['subject_entity'] and relation['subject_entity']['entity_type'] != None:
                        relation_str += relation['subject_entity']['entity_type'] + "), "
                    else:
                        relation_str += "),"
                else:
                    relation_str += ",),"
                
                if 'relation_phrase' in relation and relation['relation_phrase'] != None:
                    relation_str += relation['relation_phrase']+ ", ("
                else:
                    relation_str += ", ("
                
                if 'object_entity' in relation and relation['object_entity'] != None:
                    if 'entity_name' in relation['object_entity'] and relation['object_entity']['entity_name'] != None:
                        relation_str += relation['object_entity']['entity_name'] + ", "
                    else:
                        relation_str += ","
                    if 'entity_type' in relation['object_entity'] and relation['object_entity']['entity_type'] != None:
                        relation_str += relation['object_entity']['entity_type'] + ")); "
                    else:
                        relation_str += "));"
                else:
                    relation_str += ",));"

        #if relation and 'relation_phrase' in relation and 'subject_entity' in relation and 'object_entity' in relation:
            #relation_str += ("((" + relation['subject_entity']['entity_name'] + ", " + relation['subject_entity']['entity_type']+ "), " +
                                #relation['relation_phrase']+ ", " +
                                #"(" + relation['object_entity']['entity_name'] + ", " + relation['object_entity']['entity_type']+ "))" + "; ")
            
    return relation_str
# format_relation(train_dataset[0]["relations"])

def formatting_func(data_point):
    full_prompt = f"""Given an input text sentence, extract fact relations.
    Each fact relation describes a scientific observation or hypothesis and is in the format of a triple connecting two entities via a relation phrase: (subject_entity, relation_phrase, object_entity).
    Each subject_entity or object_entity is a chemical compound or gene/protein and is in the format of a 2-tuple: (entity_name, entity_type). Depending on the type of the entity, the entity_type must be one of ['Chemical', 'Gene/Protein'].
    The relation_phrase must be one of the following: ['increases', 'decreases', 'affects', 'binds'].
    The extracted relations should be a semicolon-separated list of relations in the format of triples: ((entity_name, entity_type), relation_phrase, (entity_name, entity_type)).
    
    ### Input sentence:
    {data_point["input_sentence"]}

    ### Extracted relations:
    {format_relation(data_point["relations"])}
    """
    return full_prompt


print(formatting_func(train_dataset[2]))
"""  
Given an input text sentence, extract fact relations.
    Each fact relation describes a scientific observation or hypothesis and is in the format of a triple connecting two entities via a relation phrase: (subject_entity, relation_phrase, object_entity).
    Each subject_entity or object_entity is a chemical compound or gene/protein and is in the format of a 2-tuple: (entity_name, entity_type). Depending on the type of the entity, the entity_type must be one of ['Chemical', 'Gene/Protein'].
    The relation_phrase must be one of the following: ['increases', 'decreases', 'affects', 'binds'].
    The extracted relations should be a semicolon-separated list of relations in the format of triples: ((entity_name, entity_type), relation_phrase, (entity_name, entity_type)).
    
    ### Input sentence:
    The results show that exposure to arsenite, increased miR-15b levels and induced M2 polarization of THP-1 cells.

    ### Extracted relations:
    ((arsenite, Chemical), increases, (miR-15b, Gene/Protein)); 
"""


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#base_model_id = "/mnt/sdc/llama_hf/llama-2-7b-hf"
base_model_id = "/home/qyfeng/llama_hf/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map='auto')
#model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map={'':torch.cuda.current_device()})

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

untokenized_text = tokenizer.decode(tokenized_train_dataset[0]['input_ids']) 
print(untokenized_text)
"""  
<|begin_of_text|>Given an input text sentence, extract fact relations.
    Each fact relation describes a scientific observation or hypothesis and is in the format of a triple connecting two entities via a relation phrase: (subject_entity, relation_phrase, object_entity).
    Each subject_entity or object_entity is a chemical compound or gene/protein and is in the format of a 2-tuple: (entity_name, entity_type). Depending on the type of the entity, the entity_type must be one of ['Chemical', 'Gene/Protein'].
    The relation_phrase must be one of the following: ['increases', 'decreases', 'affects', 'binds'].
    The extracted relations should be a semicolon-separated list of relations in the format of triples: ((entity_name, entity_type), relation_phrase, (entity_name, entity_type)).
    
    ### Input sentence:
    After treatment with 0.001, 0.01, 0.1, or 1.0 mug/mL of BPA for 48 hours, the SGK1, ENaCalpha, and phospho-SGK1 protein expression of Ishikawa cells was down-regulated, and the effect of BPA on SGK1 could be abrogated by fulvestrant.

    ### Extracted relations:
    ((BPA, Chemical), decreases, (SGK1, Gene/Protein)); ((BPA, Chemical), decreases, (ENaCalpha, Gene/Protein)); ((BPA, Chemical), decreases, (phospho-SGK1, Gene/Protein)); 
"""

import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(max(lengths))

    # Plotting the histogram
    plt.figure(figsize=(5, 3))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)


max_length = 477 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

eval_example = {'input_sentence': 'We report that three out of six treated patients developed severe hypercalcaemia following administration of ATRA, which was accompanied by a significant rise in serum IL-6 levels.',
                'relations': []}
eval_prompt = formatting_func(eval_example)
print(eval_prompt)
"""  
Given an input text sentence, extract fact relations.
    Each fact relation describes a scientific observation or hypothesis and is in the format of a triple connecting two entities via a relation phrase: (subject_entity, relation_phrase, object_entity).
    Each subject_entity or object_entity is a chemical compound or gene/protein and is in the format of a 2-tuple: (entity_name, entity_type). Depending on the type of the entity, the entity_type must be one of ['Chemical', 'Gene/Protein'].
    The relation_phrase must be one of the following: ['increases', 'decreases', 'affects', 'binds'].
    The extracted relations should be a semicolon-separated list of relations in the format of triples: ((entity_name, entity_type), relation_phrase, (entity_name, entity_type)).
    
    ### Input sentence:
    We report that three out of six treated patients developed severe hypercalcaemia following administration of ATRA, which was accompanied by a significant rise in serum IL-6 levels.

    ### Extracted relations:
"""

# LoRA
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator
model = accelerator.prepare_model(model)
# trainable params: 22030336 || all params: 4562630656 || trainable%: 0.4828428523143645


import wandb, os
wandb.login()

wandb_project = "CTD_RE_V1-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

import transformers
from datetime import datetime

project = "CTD_RE_V1-finetune"
base_model_name = "llama3-8b"
parameters = 'r_8_la_32-prompt_v3-random_2000-claude_langchain'
run_name = base_model_name + "-" + project + "-" + parameters
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=4000,
        learning_rate=2.5e-5,
        logging_steps=10,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=10,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=10,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()