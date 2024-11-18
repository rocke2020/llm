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

descriptions = res_dic["hits"]["hits"][0]["_source"]["description"]["Description"]
description_raw = "\nThe description information:\n" + "\n".join(descriptions)
ic(len(claims_raw), len(description_raw))
# 文本预处理(方便之后的关键词匹配)
claims1 = re.sub("\u2003", " ", claims_raw)
claims_real = re.sub("SEQ ID N[oO][.: ]+", "SEQ_ID_NO_", claims1)
description1 = re.sub("\u2003", " ", description_raw)
description_real = re.sub("SEQ ID N[oO][.: ]+", "SEQ_ID_NO_", description1)
print(f"{claims_real[-200:] = }")
# claim和description存储在不同的document中


target = "malassezia"
ic(len(claims), len(descriptions))
for claim in claims:
    if target in claim.lower():
        print('***claim' + claim)
        print()
print()
for description in descriptions:
    if target in description.lower():
        print('***description' + description)
        print()
