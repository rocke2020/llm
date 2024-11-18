from pandas import Series, read_json

CTD_RE_V1 = read_json("../label_studio/export/CTD_RE_v1.json").set_index("id")

# extract input text sentences
sentences = Series(data=[row["text"] for row in CTD_RE_V1.data], index=CTD_RE_V1.index)


def parse_relation_output(relation):
    """given a relation in gpt output format, returns the relation as a 5-list"""
    return [
        relation["subject_concept"].lower(),
        relation["subject_attribute"].lower(),
        relation["relation_phrase"].lower(),
        relation["object_concept"].lower(),
        relation["object_attribute"].lower(),
    ]


def parse_output(output):
    result = dict()
    result["fact_tuples"] = [parse_relation_output(r) for r in output["fact_tuples"]]
    result["condition_tuples"] = [
        parse_relation_output(r) for r in output["condition_tuples"]
    ]
    return result


task_id = 21100
input_text = sentences[task_id]
from json import load

with open("output_gpt/task" + str(task_id) + "_gpt_annotation.json") as json_file:
    gpt_output = load(json_file)
print(input_text)
parse_output(gpt_output)


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


from itertools import combinations
from re import IGNORECASE, escape, finditer
from statistics import mean

from networkx import Graph, shortest_path_length


def wordspan_distance(G, nodes_combination):
    """Given dependency graph G and a node combination, calculate the distance between the nodes"""
    n_pair = 0
    total_dist = 0
    for pair in combinations(
        [phrase for phrase in nodes_combination if phrase != []], 2
    ):
        # distance between a pair of phrases is calculated as the mean length of the shortest path between nodes (u, v)
        # where u and v are nodes contained in the two phrases respectively
        total_dist += mean(
            [
                shortest_path_length(G, source=u, target=v)
                for u in pair[0]
                for v in pair[1]
            ]
        )
        n_pair += 1
    return total_dist / n_pair


def index_in_text(input_text, relation):
    """Given a relation, returns the index of the first character the in the sentence"""
    # TODO: add cases where words are not continuously presented or not found in the sentences

    # find all occurences of the word spans
    wordspan_idx = []
    multi_occ = False
    for wordspan in relation:
        if wordspan == "":
            wordspan_idx.append([])
        else:
            occurances = [
                m.start() for m in finditer(escape(wordspan), input_text, IGNORECASE)
            ]
            if len(occurances) == 0:
                # print("Word span not be found in input text")
                return None
            else:
                wordspan_idx.append(occurances)
                if len(occurances) > 1:
                    multi_occ = True

    # return if all word spans are unique in the input text
    if not multi_occ:
        return [
            (
                (relation[i], wordspan_idx[i][0])
                if wordspan_idx[i] != []
                else (relation[i], None)
            )
            for i in range(5)
        ]
        # return [(relation[idx], idx[0]) if idx!=[] else None for idx in wordspan_idx]

    # when there are multiple occurences, construct dependency tree to find the matching occurance
    else:
        sentence = client.annotate(input_text).sentence[0]
        # extract dependency parse
        dp = sentence.basicDependencies
        # helper dictionary to associate word tokens with node labels
        # {(word, beginChar): node_label}
        token_dict = {
            (
                sentence.token[i].word.lower(),
                sentence.token[i].beginChar,
            ): sentence.token[i].tokenEndIndex
            for i in range(len(sentence.token))
        }

        # build the dependency graph
        G = Graph()
        for i in range(len(dp.edge)):
            G.add_edge(dp.edge[i].source, dp.edge[i].target)

        # find node labels for word tokens contained in each occurence of the phrases
        wordspan_nodes = []
        for i in range(5):
            wordspan = relation[i]
            if wordspan == "":
                wordspan_nodes.append([[]])
            else:
                tokenized_wordspan = [
                    t.word for t in client.annotate(wordspan).sentence[0].token
                ]
                multi_occ_nodes = []
                for idx in wordspan_idx[i]:
                    if (tokenized_wordspan[0], idx) in token_dict:
                        multi_occ_nodes.append(
                            list(
                                range(
                                    token_dict[(tokenized_wordspan[0], idx)],
                                    token_dict[(tokenized_wordspan[0], idx)]
                                    + len(tokenized_wordspan),
                                )
                            )
                        )
                if multi_occ_nodes == []:
                    wordspan_nodes.append([[]])
                else:
                    wordspan_nodes.append(multi_occ_nodes)
                # wordspan_nodes.append([list(range(i, i+len(tokenized_wordspan))) for i in [token_dict[(tokenized_wordspan[0], idx)] for idx in wordspan_idx[i]]])
        # choose the node labels that are closest to each other
        shortest_dist = 100000
        chosen_combination = []
        for nodes_combination in [
            [sc, sa, rp, oc, oa]
            for sc in wordspan_nodes[0]
            for sa in wordspan_nodes[1]
            for rp in wordspan_nodes[2]
            for oc in wordspan_nodes[3]
            for oa in wordspan_nodes[4]
        ]:
            # print(nodes_combination)
            dist = wordspan_distance(G, nodes_combination)
            # print(dist)
            if dist < shortest_dist:
                shortest_dist = dist
                chosen_combination = nodes_combination
        chosen_index = [
            sentence.token[nodes[0] - 1].beginChar if nodes != [] else None
            for nodes in chosen_combination
        ]
        indexed_phrase = []
        for i in range(5):
            if relation[i] != "" and chosen_index[i] == None:
                return None
            else:
                indexed_phrase.append((relation[i], chosen_index[i]))
        return indexed_phrase


index_in_text(input_text, parse_output(gpt_output)["fact_tuples"][0])


from uuid import uuid4


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


def extract_annotation(
    input_text, relation, relation_type, phrase_ids, labels_dict, relations_dict
):
    indexed_relation = index_in_text(input_text, relation)
    if indexed_relation != None:
        label_types = ["Concept", "Attribute", relation_type, "Concept", "Attribute"]
        for i in range(5):
            if relation[i] != "":
                # create a unique id if the phrase is not previously labelled
                if indexed_relation[i] not in phrase_ids:
                    phrase_ids[indexed_relation[i]] = uuid4().hex[:10]

                # create labels
                if indexed_relation[i] not in labels_dict:
                    labels_dict[indexed_relation[i]] = [label_types[i]]
                else:
                    labels_dict[indexed_relation[i]].append(label_types[i])

        # create annotation for each relation
        # subject attribute
        if relation[0] != "" and relation[1] != "":
            if (indexed_relation[1], indexed_relation[0]) not in relations_dict:
                relations_dict[(indexed_relation[1], indexed_relation[0])] = [
                    "attribute"
                ]
            else:
                relations_dict[(indexed_relation[1], indexed_relation[0])].append(
                    "attribute"
                )
        # object attribute
        if relation[3] != "" and relation[4] != "":
            if (indexed_relation[4], indexed_relation[3]) not in relations_dict:
                relations_dict[(indexed_relation[4], indexed_relation[3])] = [
                    "attribute"
                ]
            else:
                relations_dict[(indexed_relation[4], indexed_relation[3])].append(
                    "attribute"
                )
        # is_subject_of relation
        if relation[0] != "":
            if (indexed_relation[0], indexed_relation[2]) not in relations_dict:
                relations_dict[(indexed_relation[0], indexed_relation[2])] = ["subject"]
            else:
                relations_dict[(indexed_relation[0], indexed_relation[2])].append(
                    "subject"
                )
        # is_object_of relation
        if relation[3] != "":
            if (indexed_relation[3], indexed_relation[2]) not in relations_dict:
                relations_dict[(indexed_relation[3], indexed_relation[2])] = ["object"]
            else:
                relations_dict[(indexed_relation[3], indexed_relation[2])].append(
                    "object"
                )
    return phrase_ids, labels_dict, relations_dict


# Transform GPT output to LabelStudio input
def gpt_to_labelStudio(input_text, gpt_output):
    task = dict()
    task["data"] = {"text": input_text}

    # extract labels and relations for annotation
    phrase_ids = dict()
    labels_dict = dict()
    relations_dict = dict()
    for relation in gpt_output["fact_tuples"]:
        phrase_ids, labels_dict, relations_dict = extract_annotation(
            input_text,
            parse_relation_output(relation),
            "Fact_Relation",
            phrase_ids,
            labels_dict,
            relations_dict,
        )
    for relation in gpt_output["condition_tuples"]:
        phrase_ids, labels_dict, relations_dict = extract_annotation(
            input_text,
            parse_relation_output(relation),
            "Condition_Relation",
            phrase_ids,
            labels_dict,
            relations_dict,
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
        result.append(
            create_relation(phrase_ids[pair[0]], phrase_ids[pair[1]], list(set(labels)))
        )
    task["predictions"] = [{"result": result}]
    return task


# perform on gpt outputs
from json import dumps, load

for task_id in range(21100, 21200):
    input_text = sentences[task_id]
    with open("output_gpt/task" + str(task_id) + "_gpt_annotation.json") as json_file:
        gpt_output = load(json_file)
    lsFile = gpt_to_labelStudio(sentences[task_id], gpt_output)
    filename = "import_label_studio/task" + str(task_id) + "_annotation.json"
    with open(filename, "w") as outfile:
        outfile.write(dumps(lsFile, indent=4))


# Shut down the background CoreNLP server
client.stop()
