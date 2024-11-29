import json
import os
import random
import re
import sys

import pandas as pd
from elasticsearch import Elasticsearch
from icecream import ic
from llama_index.core import Document

sys.path.append(os.path.abspath("."))
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
from utils_llama_index.api_client import generate_api

df_seq = pd.read_csv("app/data/invention_processed.csv")
patent_kind = "A1"  # B
patent_num = (
    "20200277334"  # "09169290"  09273096 11684559  09265709  20220062139 10329336
)
patent_country = "US"


def shuffle_seq(seq_list):
    # 用于随机排列seq
    random.shuffle(seq_list)
    return "".join(seq_list)


# 对于sequence数量过多的patent，将其按一定间隔分段输入prompt
def cut_prompts(df, total_length=1000, max_num=30, rand_num=5):
    # 设置最大prompt长度和最大seq数量，防止prompt占用过多token
    prompts = []
    seq_set = []
    len_now = 0
    num_now = 0
    for index, row in df.iterrows():
        new_str = "{}:{}".format(row["new_seq_id"], row["peptide sequence"])
        # 有一项不满足就新增prompt,并将一组seq随机排列多次
        if len_now + len(new_str) > total_length or num_now + 1 > max_num:
            prompts.append([shuffle_seq(seq_set) for i in range(rand_num)])
            seq_set = []
            len_now = 0
            num_now = 0

        len_now += len(new_str)
        num_now += 1
        seq_set.append(new_str)

    prompts.append([shuffle_seq(seq_set) for i in range(rand_num)])
    return prompts


df_unit = df_seq[df_seq["patent_no"] == patent_country + patent_num + patent_kind]
print(len(df_unit))
prompts = cut_prompts(df_unit, max_num=1, rand_num=1)
for p in prompts:
    print(p)


es = Elasticsearch(hosts="http://192.168.1.124:29200")
INDEX = "patents"

body = {
    "query": {
        "bool": {
            "must": [
                {"term": {"publication_doc_number": patent_num}},
                {"term": {"publication_country": patent_country}},
                {"term": {"publication_kind": patent_kind}},
            ]
        }
    }
}

result = es.search(index=INDEX, body=body, request_timeout=60)
# 从es中获取原文
print(type(result))
res_dic = dict(result)
res_js = json.dumps(dict(result), indent=2)

print(res_dic.keys())
print(res_dic["hits"].keys())
print(res_dic["hits"]["hits"][0].keys())
print("hits num", len(res_dic["hits"]["hits"]))

soucre_keys = res_dic["hits"]["hits"][0]["_source"].keys()
print(f"{soucre_keys = }")
abstract = res_dic["hits"]["hits"][0]["_source"]["abstract"]
print(f"{abstract = }")
# dict_keys(['Reference', 'Summary', 'Description', 'Others'])
description_keys = res_dic["hits"]["hits"][0]["_source"]["description"].keys()
print(f"{description_keys = }")

summary = res_dic["hits"]["hits"][0]["_source"]["description"]["Summary"]
print("\nsummary ", len(summary), type(summary), len(summary[0]))
print(summary[0][:500])
description = res_dic["hits"]["hits"][0]["_source"]["description"]["Description"]
print("\ndescription ", len(description), type(description))
desc_lens = []
for descr in description:
    desc_lens.append(len(descr))
print(f"{desc_lens = }")
print(description[0][:500])

print()
claim = res_dic["hits"]["hits"][0]["_source"]["claim"]
print(f'{len(claim) = }, {len(claim[0]) = }\n{claim[0]["claim_text"][0][:500] = }')
print(f'{claim[-1]["claim_text"][-1][-200:] =  }')
max = 2
count = 0
for c in claim:
    # print(f'{type(c["claim_text"]) = }{len(c["claim_text"]) = }')
    for c_text in c["claim_text"]:
        if "malassezia" in c_text.lower():
            count += 1
            if count > max:
                break
            print(c_text)
print(f"{count = }")

max = 22
count = 0
claim_nums = ["153", "155", "156"]
for c in claim:
    for c_text in c["claim_text"]:
        for c_num in claim_nums:
            if c_num in c_text.lower():
                count += 1
                if count > max:
                    break
                print(c_text)
print(f"{count = }")


claims = [
    "claim " + " ".join(dic["claim_text"])
    for dic in res_dic["hits"]["hits"][0]["_source"]["claim"]
]
print(f"{len(claims) = }")
print(f"{claims[0][: 500] = }")
print(f"{claims[-1][-200:] = }")
claims_raw = "The claim information:\n" + "\n".join(claims)
description_raw = "\nThe description information:\n" + "\n".join(
    res_dic["hits"]["hits"][0]["_source"]["description"]["Description"]
)
ic(len(claims_raw), len(description_raw))
# 文本预处理(方便之后的关键词匹配)
claims1 = re.sub("\u2003", " ", claims_raw)
claims_real = re.sub("SEQ ID N[oO][.: ]+", "SEQ_ID_NO_", claims1)
description1 = re.sub("\u2003", " ", description_raw)
description_real = re.sub("SEQ ID N[oO][.: ]+", "SEQ_ID_NO_", description1)
print(f"{claims_real[-200:] = }")
# claim和description存储在不同的document中
doc_claim = Document(text=claims_real)
doc_des = Document(text=description_real)


from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter

# parse nodes
# parser = SentenceSplitter(chunk_size=256,chunk_overlap=8) #控制每个切片的长度以及相邻切片的重叠长度
# nodes = parser.get_nodes_from_documents([doc_claim,doc_des])

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512, 256])
nodes = node_parser.get_nodes_from_documents([doc_claim, doc_des])  # documents

from llama_index.core.storage import StorageContext

# define storage context
from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
# insert nodes into docstore
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)


from llama_index.core.node_parser import (
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)

print(f"{len(nodes) = }")
leaf_nodes = get_leaf_nodes(nodes)
print(f"{len(leaf_nodes) = }")
mid_nodes = get_deeper_nodes(nodes)
print(f"{len(mid_nodes) = }")
root_nodes = get_root_nodes(nodes)
print(f"{len(root_nodes) = }")


## Load index into vector index
from llama_index.core import Settings, VectorStoreIndex

Settings.embed_model = "local:/mnt/nas1/models/BAAI/bge-small-en-v1.5"

# 利用叶子节点计算相似度，并关联到对应的父节点
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

from typing import List

# import QueryBundle
from llama_index.core import QueryBundle

# Retrievers
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever

# import NodeWithScore
from llama_index.core.schema import BaseNode, NodeWithScore
from nltk import word_tokenize


class CustomRetriever(BaseRetriever):
    """自定义的混合索引类：相似度索引+关键词索引"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        vector_retriever_large: VectorIndexRetriever,
        max_num_keyword_nodes=3,
        keywords=[],
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever  # 头部相似度节点
        self._vector_retriever_large = (
            vector_retriever_large  # 更大范围的头部相似度节点(用于关键词搜索)
        )

        self.keywords = keywords  # 索引依据的关键词
        self.max_num_keyword_nodes = max_num_keyword_nodes  # 设置最大关键词节点数量
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # 利用两个不同的参数的retriever进行retrieve
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        vector_nodes_large = self._vector_retriever_large.retrieve(query_bundle)

        # 确保集合中的节点id对应的相似度由大到小排列
        vector_ids = {
            n.node_id
            for n in sorted(vector_nodes, key=lambda node: node.score, reverse=True)
        }
        # vector_ids_large = {n.node_id for n in vector_nodes_large}

        # 对于更大范围的相似度索引结果，取出其中含有关键词的节点
        keyword_ids = []
        for n in sorted(vector_nodes_large, key=lambda node: node.score, reverse=True):
            for k in self.keywords:
                if k in word_tokenize(n.get_content()):
                    # 判断关键词是否在文章片段的分词结果中
                    keyword_ids.append(n.node_id)
                    break

        combined_dict = {n.node_id: n for n in vector_nodes}
        combined_dict.update(
            {n.node_id: n for n in vector_nodes_large if n.node_id in keyword_ids}
        )

        # 合并两组节点
        if keyword_ids == []:
            # 不含有关键词的情况下照常进行相似度索引
            retrieve_ids = vector_ids
        else:
            keyword_ids_top = set(
                keyword_ids[: self.max_num_keyword_nodes]
            )  # 取相似度最高的几个关键词节点
            vector_ids_unique = (
                vector_ids - keyword_ids_top
            )  # top相似度集合中独有的节点
            retrieve_ids = keyword_ids_top  # 关键词集合和top相似度集合共有的节点+关键词集合中独有的节点
            add_num = len(vector_ids) - len(keyword_ids_top)
            retrieve_ids = set(list(vector_ids_unique)[:add_num]).union(
                retrieve_ids
            )  # 额外添加部分top相似度集合中独有的节点

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


from llama_index.core.retrievers.auto_merging_retriever import AutoMergingRetriever

base_retriever = base_index.as_retriever(similarity_top_k=6)
large_retriever = base_index.as_retriever(similarity_top_k=len(leaf_nodes) // 2)

print(f"{prompts[15][0] = }")
keywords = [i.split(":")[0] for i in prompts[15][0].split("\n")]
print(f"{keywords = }")
custom_retriever = CustomRetriever(
    base_retriever,
    large_retriever,
    max_num_keyword_nodes=3,
    keywords=[i.split(":")[0] for i in prompts[15][0].split("\n")],
)
# 创建混合索引实例
retriever = AutoMergingRetriever(custom_retriever, storage_context, verbose=True)


def query(question):
    response_nodes = retriever.retrieve(question)
    context = "\n".join(
        [
            "Context {}:\n".format(index) + node.text
            for index, node in enumerate(response_nodes)
        ]
    )
    input_str = (
        f"Context information is below.\n---------------------\n"
        f"{context}\n--------------------\n"
        f"Given the context information and not prior knowledge, answer the query:\n"
        f"{question}"
    )
    response = generate_api(input_str)
    print(response)


question1 = "Is this patent describing agents (e.g. molecules) for inhibiting Malassezia fungus? Such applications should be claimed in the “Claims” of the patent. Please make sure the patent is specific about inhibition of Malassezia. Finally, plesae answer 'Yes' or 'No' first, then explain the reason in next line."
query(question1)

question2 = """Given content describing the functions and applications of the patent.
        Please determine if sequence number "{0}" have the ability to inhibit Malassezia fungus. 
        Make sure the target sequence numbers are specific about inhibition of Malassezia and don't make any guesses.
        Please focuse on the given context and don't use prior knowledge. You should focus on sequence number "{0}" and ignore other irrelvant sequence number!
        Finally, plesae answer 'Yes' or 'No' first, then explain the reason in next line.
        """.format(
    prompts[15][0]
)
query(question2)
