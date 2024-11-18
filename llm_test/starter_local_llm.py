import os
import sys

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

sys.path.append(os.path.abspath("."))
import logging
import sys

from llm_test.local_llm_Calme import load_llama_cpp_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model(
    "local:/mnt/nas1/models/BAAI/bge-small-en-v1.5"
)

# ollama
# Settings.llm = Ollama(model="mistral", request_timeout=30.0)
Settings.llm = load_llama_cpp_model()


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "/mnt/nas1/models/llama_index_cache/quick_starter_index_storage"
store_file = PERSIST_DIR + '/docstore.json'
if not os.path.exists(store_file):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
