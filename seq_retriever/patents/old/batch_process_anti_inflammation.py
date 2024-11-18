import pandas as pd
import sys
from IPython.display import Markdown, display
import torch
from llama_index import SimpleDirectoryReader
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)

from llama_index.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)

import time
import os
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage import StorageContext
from llama_index.node_parser import get_leaf_nodes, get_root_nodes
from llama_index import VectorStoreIndex
# from llama_index.llms import LLM
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.schema import BaseNode

from typing import List
from nltk import word_tokenize 
from llama_index.llms import ChatMessage

import json
import re
from llama_index.schema import Document

#设置system_prompt
system_str=  """You are a biologist and patent expert. 
    You will be provided with some contents from a patent and will be asked to answer specific questions related to the patent. 
    Please answer the question only using the provided contents and do not make up the answer with prior knowledge."""

def load_model_loc(model_path="/home/cymei/llama_hf/llama-2-chat-7b-hf"):
    import logging
    from llama_index.llms import HuggingFaceLLM
    from llama_index.prompts import PromptTemplate

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    selected_model=model_path

    SYSTEM_PROMPT = system_str
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=selected_model,
        model_name=selected_model,
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    )

    return llm

def load_api(model="mixtral-8x7b-instruct"):
    import os

    os.environ["http_proxy"] = "http://192.168.1.254:7890"
    os.environ["https_proxy"] = "http://192.168.1.254:7890"

    from llama_index.llms.perplexity import Perplexity
    pplx_api_key = "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"
    llm = Perplexity(
        api_key=pplx_api_key, model=model, temperature=0.0,
        api_base="https://api.perplexity.ai",
    )
    return llm

class CustomRetriever(BaseRetriever):
    """自定义的混合索引类：相似度索引+关键词索引"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        vector_retriever_large: VectorIndexRetriever,
        max_num_keyword_nodes=3,
        keywords=[]
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever #头部相似度节点
        self._vector_retriever_large = vector_retriever_large #更大范围的头部相似度节点(用于关键词搜索)

        self.keywords=keywords #索引依据的关键词
        self.max_num_keyword_nodes=max_num_keyword_nodes #设置最大关键词节点数量
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        #利用两个不同的参数的retriever进行retrieve
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        vector_nodes_large = self._vector_retriever_large.retrieve(query_bundle)

        #确保集合中的节点id对应的相似度由大到小排列
        vector_ids = {n.node_id for n in sorted(vector_nodes,key=lambda node: node.score,reverse=True)} 
        # vector_ids_large = {n.node_id for n in vector_nodes_large}

        #对于更大范围的相似度索引结果，取出其中含有关键词的节点
        keyword_ids = []
        for n in sorted(vector_nodes_large,key=lambda node: node.score,reverse=True):
            for k in self.keywords:
                if(k in word_tokenize(n.get_content())):
                    #判断关键词是否在文章片段的分词结果中
                    keyword_ids.append(n.node_id)
                    break

        combined_dict = {n.node_id: n for n in vector_nodes}
        combined_dict.update({n.node_id: n for n in vector_nodes_large if n.node_id in keyword_ids })

        #合并两组节点
        if(keyword_ids==[]):
            #不含有关键词的情况下照常进行相似度索引
            retrieve_ids = vector_ids
        else:
            keyword_ids_top=set(keyword_ids[:self.max_num_keyword_nodes]) #取相似度最高的几个关键词节点
            vector_ids_unique=vector_ids-keyword_ids_top  #top相似度集合中独有的节点
            retrieve_ids=keyword_ids_top #关键词集合和top相似度集合共有的节点+关键词集合中独有的节点
            add_num=len(vector_ids)-len(keyword_ids_top)
            retrieve_ids=set(list(vector_ids_unique)[:add_num]).union(retrieve_ids) #额外添加部分top相似度集合中独有的节点

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
    

#创建RAG工作流的类
class LLM_summary:
    def __init__(self,documents=[],keywords=[]):
        #用于检索的关键词
        self.keywords=keywords
        #载入专利文本
        self.documents=documents
    
    def hierarchy_node_index(self,chunk_list=[1024, 512, 256],similarity_top_k=6,max_num_keyword_nodes=3):
        '''
        获取分层节点，并建立对应索引
        设置每一层的token数量、选入prompt的节点数量以及最多含有关键词的节点数量
        '''
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_list)
        #获取分段后的每个节点
        self.nodes = node_parser.get_nodes_from_documents(self.documents)

        #建立各个节点的存储
        docstore = SimpleDocumentStore()
        # insert nodes into docstore
        docstore.add_documents(self.nodes)
        # define storage context (will include vector store by default too)
        storage_context = StorageContext.from_defaults(docstore=docstore)
        self.leaf_nodes = get_leaf_nodes(self.nodes)
    
        # 利用叶子节点计算相似度，并关联到对应的父节点
        base_index = VectorStoreIndex(
            self.leaf_nodes,
            storage_context=storage_context,
        )

        #用于寻找和问题最相近的头部文本片段
        base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
        #用于从更大范围中寻找含有关键词的文本片段
        large_retriever = base_index.as_retriever(similarity_top_k=len(self.leaf_nodes))
        #构建混合索引
        custom_retriever = CustomRetriever(base_retriever, large_retriever,max_num_keyword_nodes=max_num_keyword_nodes,keywords=self.keywords)

        retriever = AutoMergingRetriever(custom_retriever, storage_context, verbose=True)

        return retriever 

#专利文本载入函数
#获取claim和description的信息并创建对应的document实例
def load_doc(file_name):
    with open(file_name,'r') as f:
        js_list=json.load(f)
        for js in js_list:
            if(js['article_type']=='claim'):
                claims_raw='The claim information:\n'+js['text']
            elif(js['article_type']=='description'):
                description_raw='The description information:\n'+js['text']
    
    #文本预处理
    claims1=re.sub('\u2003',' ',claims_raw)
    claims_real=re.sub('SEQ ID N[oO][.: ]+','SEQ_ID_NO_',claims1)
    description1=re.sub('\u2003',' ',description_raw)
    description_real=re.sub('SEQ ID N[oO][.: ]+','SEQ_ID_NO_',description1)

    doc_claim = Document(text=claims_real)
    doc_des = Document(text=description_real)

    return {'claim':doc_claim,'description':doc_des}


if __name__ == '__main__':
    model_location=sys.argv[1]
    if(model_location not in ['local','api']):
        raise ValueError("Model location must be 'local' or 'api'!")
    
    if(len(sys.argv)<3):
        #若没有指定模型就用默认参数
        if(model_location=='local'):
            llm=load_model_loc()
        else:
            llm=load_api()    
    else:    
        model_name=sys.argv[1]
        if(model_location=='local'):
            llm=load_model_loc(model_name)
        else:
            llm=load_api(model_name)
    print('Load model successfully!')

    #加载embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name="/home/cymei/bge-small-en",device='cuda:2'
    )
    #设置全局服务
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model 
    )
    set_global_service_context(service_context)
    print('The service is ready!')

    #载入预处理后的patent_seq数据
    #设置成数据集所在路径
    os.chdir('/home/cymei/nlp_task/anti_inflammation')
    df_seq=pd.read_csv('./patent_seq_list_new.csv')

    df_seq['is_anti_inflammation']=0
    df_seq['model_response']=''
    df_seq['supportive_context']=''

    patent_list=df_seq['patent_no'].unique() 

    #对每一个patnet下的seq进行判断
    for patent in patent_list:

        df_unit=df_seq[df_seq['patent_no']==patent]
        file_loc='./patents_parse/{}.json'.format(patent)
        doc_dic=load_doc(file_loc)
        
        #使用llama-index建立查询引擎
        #分别从claim和des中进行检索(防止因为相似度的问题全部使用description中的信息从而遗漏claim的信息)
        summarier_claim=LLM_summary([doc_dic['claim']])
        summarier_des=LLM_summary([doc_dic['description']])

        for index,row in df_unit.iterrows():

            summarier_claim.keywords=[row['seq_no']]
            summarier_des.keywords=[row['seq_no']]

            retriever_claim=summarier_claim.hierarchy_node_index(similarity_top_k=2,max_num_keyword_nodes=2)
            retriever_des=summarier_des.hierarchy_node_index(similarity_top_k=8,max_num_keyword_nodes=3)

            seq_peptides='{}:{}'.format(row['seq_no'],row['seq_peptides'])
            question="""Given content describing the functions and applications of the patent.
            Please determine if sequence number "{0}" have the ability of anti-inflammation. 
            You should focus on sequence number "{0}" and ignore other irrelvant sequence number!
            Finally, plesae do analysis and explain the reason first, then print "The answer is: 'Yes'" or "The answer is: 'No'" in next line.
            Don't answer like "Possible" or "Maybe"! Please return the top 3 most supportive context number.
                """.format(seq_peptides)

            #分别从claim和des中进行检索，确保两部分内容都会被用上
            response_nodes_claim=retriever_claim.retrieve(question) 
            response_nodes_des=retriever_des.retrieve(question) 
            response_nodes=response_nodes_claim + response_nodes_des
            #print(question)
            context_str="""Context is :
                {}
                ---------------------
                Given the provided context, please answer the query:
                {}
                """.format('\n'.join(['Context{}:\n'.format(index) + node.text for index,node in enumerate(response_nodes)]),question)  
            messages_dict = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": context_str},
            ]
            messages = [ChatMessage(**msg) for msg in messages_dict]
            response = llm.chat(messages)

            response_ans=response.message.content
            context_list=re.findall('Context[0-9]+',response_ans)
            context_nums=set([int(re.sub('Context','',context)) for context in context_list])
            #目标seq
            condition=(df_seq['patent_no']==patent) & (df_seq['seq_no']==row['seq_no'])
            if('Yes' in response_ans):
                df_seq.loc[condition,'is_anti_inflammation']=1
            elif('No' in response_ans):
                df_seq.loc[condition,'is_anti_inflammation']=0
            else:
                #没有明确yes或no的seq认为是possible
                df_seq.loc[condition,'is_anti_inflammation']=0.5

            df_seq.loc[condition,'model_response']=response_ans
            df_seq.loc[condition,'supportive_context']='\n'.join(['Context{}:\n'.format(index) + response_nodes[index].text for index in context_nums if(index<len(response_nodes))])

            print(response)
            if(model_location=='api'):
                print(response.raw['usage'])
                time.sleep(30) #设置间隔防止api调用频率过高(主要由于perplexity对10min内接收的token数量有限制)
            print('--------------------------------------------------------------------')
            
    df_seq.to_csv('/home/cymei/nlp_task/work_handover/example.csv',index=False)  