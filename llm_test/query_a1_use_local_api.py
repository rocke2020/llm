import logging
import os
import sys

sys.path.append(os.path.abspath("."))
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

from utils_llama_index.api_client import query_with_nodes

documents = SimpleDirectoryReader("data/paul_graham").load_data()


Settings.embed_model = "local:/mnt/nas1/models/BAAI/bge-small-en-v1.5"
# Settings.chunk_size = 1024
index = VectorStoreIndex.from_documents(documents)
# Ad default, the output nodes is sorted by score
retriever = index.as_retriever(similarity_top_k=5)
# will retrieve all context from the author's life
QUESTION = "What did the author do growing up?"
response_nodes = retriever.retrieve(QUESTION)
print(f"{len(response_nodes) = }")
for node in response_nodes:
    print(node)
    print(node.get_content()[:300])
    print(f"node: {node.metadata} {node.node_id} {node.score} {node.text[:100]}")
    print(f"{len(node.text) = }, {node.text[-100:] = }")
    print()
    break

run_generate_api = 0
if not run_generate_api:
    sys.exit(0)
response = query_with_nodes(QUESTION, response_nodes)
print(response)
