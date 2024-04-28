from typing import List, Optional

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from nltk import word_tokenize


def create_vector_store_index_hierarchical_node(
    docs, chunk_list=[1024, 512, 256], as_retriever=False, similarity_top_k=5
):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_list)
    nodes = node_parser.get_nodes_from_documents(docs)

    docstore = SimpleDocumentStore()
    # insert nodes into docstore
    docstore.add_documents(nodes)
    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    leaf_nodes = get_leaf_nodes(nodes)
    base_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    if as_retriever:
        retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever
    return base_index


def create_auto_merge_query_engine(
    docs, chunk_list=[1024, 512, 256], similarity_top_k=5
):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_list)
    nodes = node_parser.get_nodes_from_documents(docs)

    docstore = SimpleDocumentStore()
    # insert nodes into docstore
    docstore.add_documents(nodes)
    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    leaf_nodes = get_leaf_nodes(nodes)
    base_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    _retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(_retriever, storage_context)
    query_engine = RetrieverQueryEngine(retriever)
    return query_engine


class CustomRetriever(BaseRetriever):
    """自定义的混合索引类：相似度索引+关键词索引

    优先关键词索引，再进行相似度索引，但是控制关键词索引的数量。只要一个关键词节点被选中，就添加该node。
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        vector_retriever_large: VectorIndexRetriever,
        keywords: List[str],
        max_num_keyword_nodes=3,
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

        # 对于更大范围的相似度索引结果，取出其中含有关键词的节点
        keyword_ids = []
        for n in sorted(vector_nodes_large, key=lambda node: node.score, reverse=True):
            for k in self.keywords:
                if k in word_tokenize(n.get_content()):
                    # 判断关键词是否在文章片段的分词结果中
                    keyword_ids.append(n.node_id)
                    break

        if not keyword_ids:
            # 不含有关键词的情况下照常进行相似度索引
            retrieve_nodes = vector_nodes
        else:
            combined_dict = {n.node_id: n for n in vector_nodes}
            vector_ids = set(combined_dict.keys())
            combined_dict.update(
                {n.node_id: n for n in vector_nodes_large if n.node_id in keyword_ids}
            )
            # 取相似度最高的几个关键词节点
            keyword_ids_top = set(keyword_ids[: self.max_num_keyword_nodes])
            # top相似度集合中独有的节点
            vector_ids_unique = vector_ids - keyword_ids_top
            add_num = len(vector_ids) - len(keyword_ids_top)
            # 额外添加部分top相似度集合中独有的节点
            retrieve_ids = set(list(vector_ids_unique)[:add_num]).union(keyword_ids_top)
            retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class AutoMergingRetrieverMixedHierarchicalNodes:
    """ 自定义的混合索引类：相似度索引+关键词索引 """
    def __init__(self, documents=[], keywords=[]):
        # 用于检索的关键词
        self.keywords = keywords
        self.documents = documents

    def hierarchy_node_index(
        self, chunk_list=[1024, 512, 256], similarity_top_k=6, max_num_keyword_nodes=3
    ):
        """
        Args:
            chunk_list: 分层节点的分段大小
            similarity_top_k: The final total nodes number which combines the keyword nodes and top similarity nodes.
            max_num_keyword_nodes: 关键词节点的最大数量.
        """
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_list)
        # 获取分段后的每个节点
        self.nodes = node_parser.get_nodes_from_documents(self.documents)

        # 建立各个节点的存储
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

        # 用于寻找和问题最相近的头部文本片段
        base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
        # 用于从所有输入nodes范围中寻找含有关键词的文本片段
        large_retriever = base_index.as_retriever(similarity_top_k=len(self.leaf_nodes))
        # 构建混合索引
        custom_retriever = CustomRetriever(
            base_retriever,
            large_retriever,
            keywords=self.keywords,
            max_num_keyword_nodes=max_num_keyword_nodes,
        )

        retriever = AutoMergingRetriever(
            custom_retriever, storage_context, verbose=True
        )

        return retriever
