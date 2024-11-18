""" The few shot prompt for the MIMO annotation task. """

# input
text = "Tetrandrine triggered LC3B expression and induced autophagy in CAL 27 cells."

# expected output
{
    "text": "Tetrandrine triggered LC3B expression and induced autophagy in CAL 27 cells.",
    "fact tuples": [
        ["Tetrandrine", "", "triggered", "LC3B", "expression"],
        ["Tetrandrine", "", "induced", "autophagy", ""],
    ],
    "condition tuples": [["autophagy", "", "in", "CAL 27 cells", ""]],
    "concept_indx": [0, 2, 6, 8, 9, 10],
    "attr_indx": [3],
    "predicate_indx": [1, 5, 7],
}

# load dataset
from pandas import Series, read_json

CTD_RE_V1 = read_json("../label_studio/export/CTD_RE_v1.json").set_index("id")
# yicong = CTD_RE_V1[CTD_RE_V1.index >= 21000]
# yicong.head(1)

# extract input text sentences
sentences = Series(data=[row["text"] for row in CTD_RE_V1.data], index=CTD_RE_V1.index)


# place a space around each non-alphanumeric characters
def seperate_by_nonalnum(text):
    spaced = ""
    for c in text:
        if c.isalnum() or c == " ":
            spaced += c
        else:
            spaced += " " + c + " "
    result = ""
    for word in spaced.split():
        result += " " + word
    return result[1:]


from typing import List

from pydantic import BaseModel, Field


class Relation(BaseModel):
    """Data model for a relation. Each property is expressed by word span extracted from the input text."""

    subject_concept: str = Field(
        description="Entity (such as chemical compounds and genes) that decribe the subject of the relation"
    )
    subject_attribute: str = Field(
        description="Attribute related to the subject entity of the relation"
    )
    relation_phrase: str = Field(description="Relation phrase")
    object_concept: str = Field(
        description="Entity (such as chemical compounds and genes) that decribe the object of the relation"
    )
    object_attribute: str = Field(
        description="Attribute related to the object entity of the relation"
    )


class MIMOFormattedOutput(BaseModel):
    """Data model for a MIMO formatted output."""

    fact_tuples: List[Relation] = Field(
        description="Relations between entities that describe facts where facts are scientific observation and hypothesis"
    )
    condition_tuples: List[Relation] = Field(
        description="Relations between entities that describes conditions such as environment and equipments providing validation supports for facts"
    )
    # concept_indx: List[int] = Field(description = "Word indexes in input text of all individual words that decribe concepts in both fact_tuples and condition_tuples.")
    # attr_indx: List[int] = Field(description = "Word indexes split by space in input sentence of all individual words that decribe attributes in both fact_tuples and condition_tuples.")
    # predicate_indx: List[int] = Field(description = "Individual word indexes split by space in input sentence of all individual words that decribe relation phrases in both fact_tuples and condition_tuples.")


mimo_examples = [
    # task 21001
    {
        "text": "Tetrandrine triggered LC3B expression and induced autophagy in CAL 27 cells.",
        "fact_tuples": [
            ["Tetrandrine", "", "triggered", "LC3B", "expression"],
            ["Tetrandrine", "", "induced", "autophagy", ""],
        ],
        "condition_tuples": [["autophagy", "", "in", "CAL 27 cells", ""]],
        #'concept_indx': [0, 2, 6, 8, 9, 10],
        #'attr_indx': [3],
        #'predicate_indx': [1, 5, 7]
    },
    # task 21002
    {
        "text": "Tetrandrine and cepharanthine induce apoptosis through caspase cascade regulation, cell cycle arrest, MAPK activation and PI3K/Akt/mTOR signal modification in glucocorticoid resistant human leukemia Jurkat T cells.",
        "fact_tuples": [
            ["Tetrandrine", "", "induce", "apoptosis", ""],
            ["cepharanthine", "", "induce", "apoptosis", ""],
        ],
        "condition_tuples": [
            ["apoptosis", "", "through", "caspase cascade regulation", ""],
            ["apoptosis", "", "through", "cell cycle arrest", ""],
            ["apoptosis", "", "through", "MAPK activation", ""],
            ["apoptosis", "", "through", "PI3K/Akt/mTOR signal modification", ""],
            [
                "caspase cascade regulation",
                "",
                "in",
                "glucocorticoid resistant human leukemia Jurkat T cells",
                "",
            ],
            [
                "cell cycle arrest",
                "",
                "in",
                "glucocorticoid resistant human leukemia Jurkat T cells",
                "",
            ],
            [
                "MAPK activation",
                "",
                "in",
                "glucocorticoid resistant human leukemia Jurkat T cells",
                "",
            ],
            [
                "PI3K/Akt/mTOR signal modification",
                "",
                "in",
                "glucocorticoid resistant human leukemia Jurkat T cells",
                "",
            ],
        ],
        #'concept_indx': [0, 2, 4, 6, 7, 8, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31],
        #'attr_indx': [],
        #'predicate_indx': [3, 5, 24]
    },
    # task 21003
    {
        "text": "OBJECTIVE: To investigate the effect of thalidomide on Annexin II (AnxA2) gene regulation in multiple myeloma cell line RPMI8226 and human microvascular endothelial cell line HMEC-1 cells in vitro, and explore the potential mechanism of thrombosis induced by thalidomide.",
        "fact_tuples": [
            ["", "", "investigate", "thalidomide", "effect"],
            ["", "", "explore", "thrombosis", "potential mechanism"],
        ],
        "condition_tuples": [
            ["effect", "", "on", "Annexin II (AnxA2) gene regulation", ""],
            [
                "Annexin II (AnxA2) gene regulation",
                "",
                "in",
                "multiple myeloma cell line RPMI8226",
                "",
            ],
            [
                "Annexin II (AnxA2) gene regulation",
                "",
                "in",
                "human microvascular endothelial cell line HMEC-1 cells",
                "",
            ],
            [
                "human microvascular endothelial cell line HMEC-1 cells",
                "",
                "in",
                "vitro",
                "",
            ],
            ["potential mechanism", "", "induced by", "thalidomide", ""],
        ],
    },
    # task 21015
    # {
    #'text': 'CYP1A2 , CYP2B6 and CYP3A4 mRNA levels were induced around 48 - , 21 - and 9 - fold , respectively , by 200 microM TB , with CYP2B6 and CYP 3A4 mRNA levels being induced around 12 - and 7 - fold , respectively , by 200 microM BHT .',
    #'fact_tuples': [['CYP1A2', 'mRNA levels', 'were induced around 48 - fold by', '200 microM TB', ''],
    # ['CYP2B6', 'mRNA levels', 'were induced around 21 - fold by', '200 microM TB', ''],
    # ['CYP3A4', 'mRNA levels', 'were induced around 9 - fold by', '200 microM TB', ''],
    # ['CYP2B6', 'mRNA levels', 'being induced around 12 - fold by', '200 microM BHT', ''],
    # ['CYP 3A4', 'mRNA levels', 'being induced around 7 - fold by', '200 microM BHT', '']],
    #'condition_tuples': [],
    #'concept_indx': [0, 2, 4, 23, 24, 25, 28, 30, 31, 47, 48, 49],
    #'attr_indx': [5, 6, 32, 33],
    #'predicate_indx': [7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 22, 34, 35, 36, 37, 38, 40, 41, 42, 46]
    # }
]


# convert mimo output to previously defined pydantic data model
def tuplesToRelation(tpl):
    return {
        "subject_concept": tpl[0],
        "subject_attribute": tpl[1],
        "relation_phrase": tpl[2],
        "object_concept": tpl[3],
        "object_attribute": tpl[4],
    }


def mimoToFewShot(mimo_example):
    fewshot_example = dict()
    fewshot_example["input text"] = mimo_example["text"]
    fewshot_example["MIMO output"] = (
        MIMOFormattedOutput.model_validate(
            {
                "fact_tuples": [
                    tuplesToRelation(t) for t in mimo_example["fact_tuples"]
                ],
                "condition_tuples": [
                    tuplesToRelation(t) for t in mimo_example["condition_tuples"]
                ],
                # "concept_indx": mimo_example["concept_indx"],
                # "attr_indx": mimo_example["attr_indx"],
                # "predicate_indx": mimo_example["predicate_indx"]
            }
        )
        .model_dump_json()
        .replace("{", "{{")
        .replace("}", "}}")
    )
    return fewshot_example


fewshot_examples = [mimoToFewShot(mimo_example) for mimo_example in mimo_examples]


# generate prompt with fewshot examples
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input text", "MIMO output"],
    template="Input text: {input text}\nMIMO output:\n{MIMO output}",
)
# print(example_prompt.format(**fewshot_examples[0]))

from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MIMOFormattedOutput)
prompt = FewShotPromptTemplate(
    examples=fewshot_examples,
    example_prompt=example_prompt,
    prefix="""Given an input text sentence, extract the fact and condition relations in the input text.
    {format_instructions}
    """,
    suffix="",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

print(
    prompt.format().replace("{", "{{").replace("}", "}}")
    + "\n\nText: {input}\nMIMO output:\n"
)


""" A good few shot example 

Given an input text sentence, extract the fact and condition relations in the input text.
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{{"$defs": {{"Relation": {{"description": "Data model for a relation. Each property is expressed by word span extracted from the input text.", "properties": {{"subject_concept": {{"description": "Entity (such as chemical compounds and genes) that decribe the subject of the relation", "title": "Subject Concept", "type": "string"}}, "subject_attribute": {{"description": "Attribute related to the subject entity of the relation", "title": "Subject Attribute", "type": "string"}}, "relation_phrase": {{"description": "Relation phrase", "title": "Relation Phrase", "type": "string"}}, "object_concept": {{"description": "Entity (such as chemical compounds and genes) that decribe the object of the relation", "title": "Object Concept", "type": "string"}}, "object_attribute": {{"description": "Attribute related to the object entity of the relation", "title": "Object Attribute", "type": "string"}}}}, "required": ["subject_concept", "subject_attribute", "relation_phrase", "object_concept", "object_attribute"], "title": "Relation", "type": "object"}}}}, "description": "Data model for a MIMO formatted output.", "properties": {{"fact_tuples": {{"description": "Relations between entities that describe facts where facts are scientific observation and hypothesis", "items": {{"$ref": "#/$defs/Relation"}}, "title": "Fact Tuples", "type": "array"}}, "condition_tuples": {{"description": "Relations between entities that describes conditions such as environment and equipments providing validation supports for facts", "items": {{"$ref": "#/$defs/Relation"}}, "title": "Condition Tuples", "type": "array"}}}}, "required": ["fact_tuples", "condition_tuples"]}}
```
    

Input text: Tetrandrine triggered LC3B expression and induced autophagy in CAL 27 cells.
MIMO output:
{{"fact_tuples":[{{"subject_concept":"Tetrandrine","subject_attribute":"","relation_phrase":"triggered","object_concept":"LC3B","object_attribute":"expression"}},{{"subject_concept":"Tetrandrine","subject_attribute":"","relation_phrase":"induced","object_concept":"autophagy","object_attribute":""}}],"condition_tuples":[{{"subject_concept":"autophagy","subject_attribute":"","relation_phrase":"in","object_concept":"CAL 27 cells","object_attribute":""}}]}}

Input text: Tetrandrine and cepharanthine induce apoptosis through caspase cascade regulation, cell cycle arrest, MAPK activation and PI3K/Akt/mTOR signal modification in glucocorticoid resistant human leukemia Jurkat T cells.
MIMO output:
{{"fact_tuples":[{{"subject_concept":"Tetrandrine","subject_attribute":"","relation_phrase":"induce","object_concept":"apoptosis","object_attribute":""}},{{"subject_concept":"cepharanthine","subject_attribute":"","relation_phrase":"induce","object_concept":"apoptosis","object_attribute":""}}],"condition_tuples":[{{"subject_concept":"apoptosis","subject_attribute":"","relation_phrase":"through","object_concept":"caspase cascade regulation","object_attribute":""}},{{"subject_concept":"apoptosis","subject_attribute":"","relation_phrase":"through","object_concept":"cell cycle arrest","object_attribute":""}},{{"subject_concept":"apoptosis","subject_attribute":"","relation_phrase":"through","object_concept":"MAPK activation","object_attribute":""}},{{"subject_concept":"apoptosis","subject_attribute":"","relation_phrase":"through","object_concept":"PI3K/Akt/mTOR signal modification","object_attribute":""}},{{"subject_concept":"caspase cascade regulation","subject_attribute":"","relation_phrase":"in","object_concept":"glucocorticoid resistant human leukemia Jurkat T cells","object_attribute":""}},{{"subject_concept":"cell cycle arrest","subject_attribute":"","relation_phrase":"in","object_concept":"glucocorticoid resistant human leukemia Jurkat T cells","object_attribute":""}},{{"subject_concept":"MAPK activation","subject_attribute":"","relation_phrase":"in","object_concept":"glucocorticoid resistant human leukemia Jurkat T cells","object_attribute":""}},{{"subject_concept":"PI3K/Akt/mTOR signal modification","subject_attribute":"","relation_phrase":"in","object_concept":"glucocorticoid resistant human leukemia Jurkat T cells","object_attribute":""}}]}}

Input text: OBJECTIVE: To investigate the effect of thalidomide on Annexin II (AnxA2) gene regulation in multiple myeloma cell line RPMI8226 and human microvascular endothelial cell line HMEC-1 cells in vitro, and explore the potential mechanism of thrombosis induced by thalidomide.
MIMO output:
{{"fact_tuples":[{{"subject_concept":"","subject_attribute":"","relation_phrase":"investigate","object_concept":"thalidomide","object_attribute":"effect"}},{{"subject_concept":"","subject_attribute":"","relation_phrase":"explore","object_concept":"thrombosis","object_attribute":"potential mechanism"}}],"condition_tuples":[{{"subject_concept":"effect","subject_attribute":"","relation_phrase":"on","object_concept":"Annexin II (AnxA2) gene regulation","object_attribute":""}},{{"subject_concept":"Annexin II (AnxA2) gene regulation","subject_attribute":"","relation_phrase":"in","object_concept":"multiple myeloma cell line RPMI8226","object_attribute":""}},{{"subject_concept":"Annexin II (AnxA2) gene regulation","subject_attribute":"","relation_phrase":"in","object_concept":"human microvascular endothelial cell line HMEC-1 cells","object_attribute":""}},{{"subject_concept":"human microvascular endothelial cell line HMEC-1 cells","subject_attribute":"","relation_phrase":"in","object_concept":"vitro","object_attribute":""}},{{"subject_concept":"potential mechanism","subject_attribute":"","relation_phrase":"induced by","object_concept":"thalidomide","object_attribute":""}}]}}

Text: {input}
MIMO output:

"""

from llama_index.llms import OpenAI

# Define openai pydantic program
from llama_index.program import OpenAIPydanticProgram

program = OpenAIPydanticProgram.from_defaults(
    output_cls=MIMOFormattedOutput,
    llm=OpenAI(model="gpt-4-1106-preview"),
    prompt_template_str=prompt.format().replace("{", "{{").replace("}", "}}")
    + "\n\nText: {input}\nMIMO output:\n",
    verbose=False,
)


from json import dumps

# annotate 100 sentences (task 21100 - 21199) and save output as json files
from tqdm import tqdm

for task_id in tqdm(range(21100, 21200)):
    with open(
        "output_gpt/task" + str(task_id) + "_gpt_annotation.json", "w"
    ) as outfile:
        outfile.write(dumps(program(input=sentences[task_id]).model_dump(), indent=4))


# parse output from openai to MIMO
def parse_relation_output(relation):
    return [
        relation["subject_concept"],
        relation["subject_attribute"],
        relation["relation_phrase"],
        relation["object_concept"],
        relation["object_attribute"],
    ]


def parse_output(input_text, output):
    result = dict()
    result["text"] = input_text
    result["fact_tuples"] = [parse_relation_output(r) for r in output["fact_tuples"]]
    result["condition_tuples"] = [
        parse_relation_output(r) for r in output["condition_tuples"]
    ]
    return result


from json import load

task_id = 21102
with open("output_gpt/task" + str(task_id) + "_gpt_annotation.json") as json_file:
    gpt_output = load(json_file)
parse_output(sentences[task_id], gpt_output)
"""  
{'text': 'The expression of HOXA9 mRNA in ATRA groups at day 1-3 was higher than that in control group (P < 0.05).',
 'fact_tuples': [['HOXA9 mRNA',
   'expression',
   'was higher',
   'ATRA groups',
   'day 1-3'],
  ['HOXA9 mRNA', 'expression', 'was higher than', 'control group', '']],
 'condition_tuples': [['expression', '', 'in', 'ATRA groups', 'day 1-3'],
  ['expression', '', 'in', 'control group', ''],
  ['expression', 'higher', 'with significance', 'P', '< 0.05']]}
"""

task_id = 21196
with open("output_gpt/task" + str(task_id) + "_gpt_annotation.json") as json_file:
    gpt_output = load(json_file)
parse_output(sentences[task_id], gpt_output)
"""  
{'text': 'Induction of apoptosis by trichostatin A, a histone deacetylase inhibitor, is associated with inhibition of cyclooxygenase-2 activity in human non-small cell lung cancer cells.',
 'fact_tuples': [['trichostatin A', '', 'induction of', 'apoptosis', '']],
 'condition_tuples': [['apoptosis',
   '',
   'is associated with',
   'inhibition',
   'of cyclooxygenase-2 activity'],
  ['inhibition',
   'of cyclooxygenase-2 activity',
   'in',
   'human non-small cell lung cancer cells',
   '']]}
"""


# parse output from openai to MIMO
def parse_relation_output2(relation):
    return [
        relation.subject_concept,
        relation.subject_attribute,
        relation.relation_phrase,
        relation.object_concept,
        relation.object_attribute,
    ]


def parse_output2(input_text, output):
    result = dict()
    result["text"] = input_text
    result["fact_tuples"] = [parse_relation_output2(r) for r in output.fact_tuples]
    result["condition_tuples"] = [
        parse_relation_output2(r) for r in output.condition_tuples
    ]
    result["concept_indx"] = output.concept_indx
    result["attr_indx"] = output.attr_indx
    result["predicate_indx"] = output.predicate_indx
    return result
