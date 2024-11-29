from time import sleep

from stanfordnlp.server import CoreNLPClient

# Construct a CoreNLPClient with annotators, a memory allocation of 4GB, and port number 9001
client = CoreNLPClient(
    annotators=["tokenize", "mwt", "pos", "lemma", "depparse"],
    pipelineLanguage="en",
    memory="4G",
    endpoint="http://localhost:9001",
)

# Start the background server and wait for some time
client.start()
sleep(5)

# load input sentences
from pandas import Series, read_json

CTD_RE_V1 = read_json("../label_studio/export/CTD_RE_v1.json").set_index("id")
sentences = Series(data=[row["text"] for row in CTD_RE_V1.data], index=CTD_RE_V1.index)

# load train and test sample indexes
from csv import reader

with open("sampled_train_ids.csv", "r") as file:
    sampled_train_ids = list(map(int, list(reader(file, delimiter=","))[0]))
    file.close()
with open("sampled_test_ids.csv", "r") as file:
    sampled_test_ids = list(map(int, list(reader(file, delimiter=","))[0]))
    file.close()

# load json output and return the relations as a list of tuples
from json import dumps, load


def json_to_tuple(relations):
    if relations != None:
        relation_tuples = []
        for relation in relations:
            if relation != None:
                if "subject_entity" in relation and relation["subject_entity"] != None:
                    if (
                        "entity_name" in relation["subject_entity"]
                        and relation["subject_entity"]["entity_name"] != None
                    ):
                        subj_entity_name = relation["subject_entity"]["entity_name"]
                    else:
                        subj_entity_name = ""
                    if (
                        "entity_type" in relation["subject_entity"]
                        and relation["subject_entity"]["entity_type"] != None
                    ):
                        subj_entity_type = relation["subject_entity"]["entity_type"]
                    else:
                        subj_entity_type = ""
                else:
                    subj_entity_name = ""
                    subj_entity_type = ""

                if (
                    "relation_phrase" in relation
                    and relation["relation_phrase"] != None
                ):
                    relation_phrase = relation["relation_phrase"]
                else:
                    relation_phrase = ""

                if "object_entity" in relation and relation["object_entity"] != None:
                    if (
                        "entity_name" in relation["object_entity"]
                        and relation["object_entity"]["entity_name"] != None
                    ):
                        obj_entity_name = relation["object_entity"]["entity_name"]
                    else:
                        obj_entity_name = ""
                    if (
                        "entity_type" in relation["object_entity"]
                        and relation["object_entity"]["entity_type"] != None
                    ):
                        obj_entity_type = relation["object_entity"]["entity_type"]
                    else:
                        obj_entity_type = ""
                else:
                    obj_entity_name = ""
                    obj_entity_type = ""
            relation_tuples.append(
                [
                    subj_entity_name,
                    subj_entity_type,
                    relation_phrase,
                    obj_entity_name,
                    obj_entity_type,
                ]
            )
    return relation_tuples


def load_output(filename):
    with open(filename) as json_file:
        output = load(json_file)
        if output and "relations" in output:
            if isinstance(output["relations"], list):
                return json_to_tuple(output["relations"])
    json_file.close()
    return []


"""  
task_id = 21503
input_text = sentences[task_id]
print(input_text)
# '../claude/no_attribute/data/random_2000/langchain_output/output_claude/' + 'task' + str(task_id) + '_claude_annotation.json'
# '../gpt/no_attribute/data/random_2000/output_gpt/' + 'task' + str(task_id) + '_gpt_annotation.json'
load_output('../claude/no_attribute/data/random_2000/langchain_output/output_claude/' + 'task' + str(task_id) + '_claude_annotation.json')

Exposed to As2O3 at 3 microM for 72 h, SK-N-SH cells exhibited lower expression of P-gp than 2 microM As2O3 for 72 h. In contrast, the expression of P-gp was upregulated by DDP and VP16.
[['As2O3', 'Chemical', 'decreases', 'P-gp', 'Gene/Protein'],
 ['DDP', 'Chemical', 'increases', 'P-gp', 'Gene/Protein'],
 ['VP16', 'Chemical', 'increases', 'P-gp', 'Gene/Protein']]

"""

from itertools import combinations
from math import inf
from re import IGNORECASE, escape, finditer
from statistics import mean

from networkx import Graph, NetworkXNoPath, shortest_path_length


def wordspan_distance(G, nodes_combination):
    """Given dependency graph G and a node combination, calculate the distance between the nodes"""
    n_pair = 0
    total_dist = 0
    for pair in combinations(
        [phrase for phrase in nodes_combination if phrase != []], 2
    ):
        try:
            # distance between a pair of phrases is calculated as the mean length of the shortest path between nodes (u, v)
            # where u and v are nodes contained in the two phrases respectively
            total_dist += mean(
                [
                    shortest_path_length(G, source=u, target=v)
                    for u in pair[0]
                    for v in pair[1]
                ]
            )
        except NetworkXNoPath:
            total_dist += inf
        n_pair += 1
    if n_pair != 0:
        return total_dist / n_pair
    return inf


def index_in_text(input_text, relation):
    """Here use the triple format of relation, not MIMO annotation format
    Given a relation, returns the index of the first character the in the sentence
    Slightly different from the "app\feature_relation_extraction\MIMO_annotation\GPT_output_to_LabelStudio_input_conversion.py" version
    """
    # TODO: add cases where words are not continuously presented or not found in the sentences

    # find all occurences of the word spans
    wordspan_idx = []
    multi_occ = False
    # relation[0], relation[3] are the subject and object entities
    relation = [relation[0], relation[3]]
    for wordspan in relation:
        if wordspan == "":
            wordspan_idx.append([])
        else:
            occurances = [
                m.start() for m in finditer(escape(wordspan), input_text, IGNORECASE)
            ]
            if len(occurances) == 0:
                # print("Word span not found in input text")
                return None
            else:
                wordspan_idx.append(occurances)
                if len(occurances) > 1:
                    multi_occ = True

    # if all word spans are unique in the input text
    if not multi_occ:
        return [
            (
                (relation[i], wordspan_idx[i][0])
                if wordspan_idx[i] != []
                else (relation[i], None)
            )
            for i in range(len(relation))
        ]

    # when there are multiple occurences, construct dependency tree to find the matching occurance
    else:
        token_node_dict = dict({})
        G = Graph()
        sentences = client.annotate(input_text).sentence
        for sentence in sentences:
            tokenOffset = sentence.tokenOffsetBegin
            # extract dependency parse
            dp = sentence.basicDependencies
            # helper dictionary to associate word tokens with node labels
            # {(word, beginChar): node_label}
            for i in range(len(sentence.token)):
                token_node_dict[
                    (sentence.token[i].word, sentence.token[i].beginChar)
                ] = sentence.token[i].tokenEndIndex
            # build the dependency graph
            for i in range(len(dp.edge)):
                G.add_edge(
                    dp.edge[i].source + tokenOffset, dp.edge[i].target + tokenOffset
                )
        # helper dictionary to associate node labels with word tokens
        # {node_label: (word, beginChar)}
        node_token_dict = {v: k for k, v in token_node_dict.items()}

        # find node labels for word tokens contained in each occurence of the phrases
        wordspan_nodes = []
        for i in range(len(relation)):
            wordspan = relation[i]
            if wordspan == "":
                wordspan_nodes.append([[]])
            else:
                tokenized_wordspan = [
                    t.word for t in client.annotate(wordspan).sentence[0].token
                ]
                multi_occ_nodes = []
                for idx in wordspan_idx[i]:
                    if (tokenized_wordspan[0], idx) in token_node_dict:
                        multi_occ_nodes.append(
                            list(
                                range(
                                    token_node_dict[(tokenized_wordspan[0], idx)],
                                    token_node_dict[(tokenized_wordspan[0], idx)]
                                    + len(tokenized_wordspan),
                                )
                            )
                        )
                if multi_occ_nodes == []:
                    wordspan_nodes.append([[]])
                else:
                    wordspan_nodes.append(multi_occ_nodes)

        # choose the node labels that are closest to each other
        shortest_dist = inf
        chosen_combination = []
        for nodes_combination in [
            [sbj, obj] for sbj in wordspan_nodes[0] for obj in wordspan_nodes[1]
        ]:
            # print(nodes_combination)
            dist = wordspan_distance(G, nodes_combination)
            # print(dist)
            if dist < shortest_dist:
                shortest_dist = dist
                chosen_combination = nodes_combination

        chosen_index = [
            node_token_dict[nodes[0]][1] if nodes != [] else None
            for nodes in chosen_combination
        ]
        indexed_phrase = []
        for i in range(len(relation)):
            if i + 1 > len(chosen_index):
                return None
            else:
                if relation[i] != "" and chosen_index[i] == None:
                    return None
                else:
                    indexed_phrase.append((relation[i], chosen_index[i]))
        return indexed_phrase


"""  
task_id = 21503
input_text = sentences[task_id]
print(input_text)
output = load_output('../claude/no_attribute/data/random_2000/langchain_output/output_claude/' + 'task' + str(task_id) + '_claude_annotation.json')
index_in_text(input_text, output[2])

Exposed to As2O3 at 3 microM for 72 h, SK-N-SH cells exhibited lower expression of P-gp than 2 microM As2O3 for 72 h. In contrast, the expression of P-gp was upregulated by DDP and VP16.
[('VP16', 181), ('P-gp', 149)]
"""


# Transform LLM output to LabelStudio input
from uuid import uuid4

from biobert_embedding.embedding import BiobertEmbedding
from scipy.spatial import distance

biobert = BiobertEmbedding(model_path="biobert_v1.1_pubmed_pytorch_model")


def create_label(text, start, labels, id):
    return {
        "value": {
            "start": start,
            "end": start + len(text),
            "text": text,
            "labels": labels,
        },
        "id": id,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
    }


def create_relation(from_id, to_id, labels):
    return {
        "from_id": from_id,
        "to_id": to_id,
        "type": "relation",
        "direction": "right",
        "labels": labels,
    }


def extract_annotation(input_text, relation, phrase_ids, labels_dict, relations_dict):
    label_type = [relation[1], relation[4]]
    indexed_relation = index_in_text(input_text, relation)
    if indexed_relation != None:
        for i in [0, 1]:
            if indexed_relation[i][0] != "":
                # create a unique id if the phrase is not previously labelled
                if indexed_relation[i] not in phrase_ids:
                    phrase_ids[indexed_relation[i]] = uuid4().hex[:10]

                # create labels for each region
                if indexed_relation[i] not in labels_dict:
                    labels_dict[indexed_relation[i]] = [label_type[i]]
                else:
                    labels_dict[indexed_relation[i]].append(label_type[i])

        # create annotation for each relation
        if relation[2] != "":
            if relation[2] in ["increases", "decreases", "affects", "binds"]:
                relation_phrase = relation[2]
            else:
                relation_phrase = "affects"
                max_sim = -1
                for word in ["increases", "decreases", "affects", "binds"]:
                    cos_sim = 1 - distance.cosine(
                        biobert.sentence_vector(relation[2]),
                        biobert.sentence_vector(word),
                    )
                    if cos_sim > max_sim:
                        max_sim = cos_sim
                        relation_phrase = word

            if (indexed_relation[0], indexed_relation[1]) not in relations_dict:
                relations_dict[(indexed_relation[0], indexed_relation[1])] = [
                    relation_phrase
                ]
            else:
                relations_dict[(indexed_relation[0], indexed_relation[1])].append(
                    relation_phrase
                )

    return phrase_ids, labels_dict, relations_dict


def llm_output_to_labelStudio(input_text, output, inner_id):
    """Given the input text and the output in the format of a list of tuples, return the label studio import format"""
    task = dict()
    task["data"] = {"text": input_text}
    if inner_id != None:
        task["inner_id"] = inner_id

    # extract labels and relations for annotation
    phrase_ids = dict()
    labels_dict = dict()
    relations_dict = dict()
    for relation in output:
        phrase_ids, labels_dict, relations_dict = extract_annotation(
            input_text, relation, phrase_ids, labels_dict, relations_dict
        )

    # create label studio formatted annotations for extracted labels and relations
    result = []
    for indexed_phrase, labels in labels_dict.items():
        result.append(
            create_label(
                indexed_phrase[0],
                indexed_phrase[1],
                list(set(labels)),
                phrase_ids[indexed_phrase],
            )
        )
    for pair, labels in relations_dict.items():
        if pair[0] in phrase_ids and pair[1] in phrase_ids:
            result.append(
                create_relation(
                    phrase_ids[pair[0]], phrase_ids[pair[1]], list(set(labels))
                )
            )
    task["predictions"] = [{"result": result}]
    return task


# Perform on annotations by Claude
ls_preannotations = []
for task_id in sampled_train_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../claude/no_attribute/data/random_2000/langchain_output/output_claude/"
        + "task"
        + str(task_id)
        + "_claude_annotation.json"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, None)
    )

for task_id in sampled_test_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../test_output_2000/claude/claude_langchain_annotation/"
        + "task"
        + str(task_id)
        + "_claude_annotation.json"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, None)
    )

filename = "data/claude_annotation.json"
with open(filename, "w") as outfile:
    outfile.write(dumps(ls_preannotations, indent=4))
    outfile.close()

# perform on gpt annotations
ls_preannotations = []
for task_id in sampled_train_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../gpt/no_attribute/data/random_2000/output_gpt/"
        + "task"
        + str(task_id)
        + "_gpt_annotation.json"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, None)
    )

for task_id in sampled_test_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../test_output_2000/gpt/gpt_annotation/"
        + "task"
        + str(task_id)
        + "_gpt_annotation.json"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, None)
    )

filename = "data/gpt_annotation.json"
with open(filename, "w") as outfile:
    outfile.write(dumps(ls_preannotations, indent=4))
    outfile.close()

# Perform on annotations by fintuned Llama3
# load llama_annotations
def format_matched_string(s):
    while s.startswith(("'", '"', "(")):
        s = s[1:]
    while s.endswith(("'", '"', ")")):
        s = s[:-1]
    return s


from re import findall


def load_output_llama3(filename):
    with open(filename, "r") as f:
        llama_output = f.read()
        f.close()
    output_start_id = llama_output.find("### Extracted relations:")
    # find pattern: (({some_text}, {some_text}), {some_text}, ({some_text}, {some_text}))
    # triple_pattern = r"\(\(([^,]+),\s*([^,]+)\),\s*([^,]+),\s*\(([^,]+),\s*([^,]+)\)\)"
    # find pattern: non greedy ({some_text}, {some_text}), {some_text}, ({some_text}, {some_text})
    triple_pattern = r"\(([^,]+?),\s*([^,]+?)\),\s*([^,]+?),\s*\(([^,]+?),\s*([^,]+?)\)"
    # TODO: problem with stripping ")" at the end of the string: TGF-beta(1
    matches = findall(triple_pattern, llama_output[output_start_id:])
    matches = [[format_matched_string(s) for s in match] for match in matches]
    return [[m[0], m[1], m[2], m[3], m[4]] for m in matches]


# perform on annotations by llama3 finetuned with claude
# similar perform on annotations by llama3 finetuned with gpt, the same code can be used
ls_preannotations = []
inner_id = 1
for task_id in sampled_test_ids:
    input_text = sentences[task_id]
    output = load_output_llama3(
        "../test_output_2000/claude/llama3-8b-CTD_RE_V1-finetune-r_8_la_32-prompt_v3-random_2000-claude_langchain-checkpoint-260/"
        + str(task_id)
        + ".txt"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, inner_id)
    )
    inner_id += 1

for task_id in sampled_test_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../test_output_2000/claude/llama3-8b-CTD_RE_V1-finetune-r_8_la_32-prompt_v3-random_2000-claude_langchain-checkpoint-450/"
        + str(task_id)
        + ".txt"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, inner_id)
    )
    inner_id += 1

for task_id in sampled_test_ids:
    input_text = sentences[task_id]
    output = load_output(
        "../test_output_2000/claude/llama3-8b-CTD_RE_V1-finetune-r_8_la_32-prompt_v3-random_2000-claude_langchain-checkpoint-670/"
        + str(task_id)
        + ".txt"
    )
    ls_preannotations.append(
        llm_output_to_labelStudio(sentences[task_id], output, inner_id)
    )
    inner_id += 1

filename = "data/llama3(claude)_annotation.json"
with open(filename, "w") as outfile:
    outfile.write(dumps(ls_preannotations, indent=4))
    outfile.close()

# Shut down the background CoreNLP server
client.stop()
