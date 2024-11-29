import logging
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import torch
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM

documents = SimpleDirectoryReader("data/paul_graham").load_data()
from llama_index.core import PromptTemplate

Settings.embed_model = "local:/mnt/nas1/models/BAAI/bge-small-en-v1.5"
Settings.chunk_size = 1024
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()
# will retrieve all context from the author's life
QUESTION = "What did the author do growing up?"
response_nodes = retriever.retrieve(QUESTION)
print(f"{len(response_nodes) = }")
for node in response_nodes:
    print(node)
    print(f"node: {node.metadata} {node.node_id} {node.score} {node.text[:100]}")
    print(f"{len(node.text) = }, {node.text[-100:] = }")
    print()
    break
# sys.exit(0)


def completion_to_prompt_llama2(completion):
    return f"<s>[INST] {completion} [/INST]"


model_type = "Starling_LM"
if model_type == "Mistral-7B-Instruct-v0.2":
    query_wrapper_prompt = PromptTemplate("<s>[INST] {query_str} [/INST]")
    model_dir = "/mnt/nas1/models/mistralai/Mistral-7B-Instruct-v0.2"
elif model_type == "Starling_LM":
    query_wrapper_prompt = PromptTemplate(
        "GPT4 Correct User: {query_str}<|end_of_turn|>GPT4 Correct Assistant:"
    )
    model_dir = "/mnt/nas1/models/Nexusflow/Starling-LM-7B-beta"

quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"do_sample": False, "pad_token_id": 2},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model_dir,
    model_name=model_dir,
    device_map="auto",
    # stopping_ids=[50278, 50279, 50277, 1, 0],
    # tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={
        "quantization_config": quantization_config_8bit,
        "torch_dtype": torch.float16,
    },
)


## llm.is_chat_model = False
run_default_query_engine = 1
if run_default_query_engine:
    print(f"{model_dir = } {llm.is_chat_model = }")
    Settings.llm = llm
    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine()
    query_engine_get_prompts = query_engine.get_prompts()
    response = query_engine.query(QUESTION)
    print(response)
    print()
    source_nodes = response.source_nodes
    node = source_nodes[0]
    print(f"node: {node.metadata} {node.node_id} {node.score} {node.text[:100]}")
    print(f"{len(node.text) = }, {node.text[-100:] = }")

    print(f'{len(response_nodes) = }, {len(source_nodes) = }')
    for node, node2 in zip(response_nodes, source_nodes):
        same = node.text == node2.text
        print(f'{same = }')
else:
    context_str = """Context is :
        {}
        ---------------------
        Given the provided context, please answer the query:
        {}
        """.format(
        "\n".join(
            [
                "Context {}:\n".format(index) + node.text
                for index, node in enumerate(response_nodes)
            ]
        ),
        QUESTION,
    )
    print(f"{context_str = }\n")
    # input = completion_to_prompt(context_str)
    response = llm.complete(context_str)
    print(response)

""" query_engine_get_prompts 

query_engine_get_prompts = {'response_synthesizer:text_qa_template': SelectorPromptTemplate(metadata={'prompt_type': <PromptType.QUESTION_ANSWER: 'text_qa'>}, template_vars=['context_str', 'query_str'], kwargs={}, output_parser=None, template_var_mappings={}, function_mappings={}, default_template=PromptTemplate(metadata={'prompt_type': <PromptType.QUESTION_ANSWER: 'text_qa'>}, template_vars=['context_str', 'query_str'], kwargs={}, output_parser=None, template_var_mappings=None, function_mappings=None, template=
'Context information is below.\n---------------------\n
{context_str}\n---------------------\n
Given the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: '),
conditionals=[(<function is_chat_model at 0x7fa6b82252d0>, ChatPromptTemplate(metadata={'prompt_type': <PromptType.CUSTOM: 'custom'>}, template_vars=['context_str', 'query_str'], kwargs={}, output_parser=None, template_var_mappings=None, function_mappings=None, message_templates=[ChatMessage(role=<MessageRole.SYSTEM: 'system'>, content="You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.", additional_kwargs={}), ChatMessage(role=<MessageRole.USER: 'user'>, content='Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: ', additional_kwargs={})]))]), 'response_synthesizer:refine_template': SelectorPromptTemplate(metadata={'prompt_type': <PromptType.REFINE: 'refine'>}, template_vars=['query_str', 'existing_answer', 'context_msg'], kwargs={}, output_parser=None, template_var_mappings={}, function_mappings={}, default_template=PromptTemplate(metadata={'prompt_type': <PromptType.REFINE: 'refine'>}, template_vars=['query_str', 'existing_answer', 'context_msg'], kwargs={}, output_parser=None, template_var_mappings=None, function_mappings=None, template="The original query is as follows: {query_str}\nWe have provided an existing answer: {existing_answer}\nWe have the opportunity to refine the existing answer (only if needed) with some more context below.\n------------\n{context_msg}\n------------\nGiven the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\nRefined Answer: "), conditionals=[(<function is_chat_model at 0x7fa6b82252d0>, ChatPromptTemplate(metadata={'prompt_type': <PromptType.CUSTOM: 'custom'>}, template_vars=['context_msg', 'query_str', 'existing_answer'], kwargs={}, output_parser=None, template_var_mappings=None, function_mappings=None, message_templates=[ChatMessage(role=<MessageRole.USER: 'user'>, content="You are an expert Q&A system that strictly operates in two modes when refining existing answers:\n1. **Rewrite** an original answer using the new context.\n2. **Repeat** the original answer if the new context isn't useful.\nNever reference the original answer or context directly in your answer.\nWhen in doubt, just repeat the original answer.\nNew Context: {context_msg}\nQuery: {query_str}\nOriginal Answer: {existing_answer}\nNew Answer: ", additional_kwargs={})]))])}
"""
