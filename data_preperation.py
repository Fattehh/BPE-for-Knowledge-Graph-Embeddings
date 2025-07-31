import os
import random
from collections import defaultdict

import pandas
import pandas as pd


def KG_selection(fine_tuned_on: str = "", included_KGs_dict: dict[str, list] = ["KGs"]) -> tuple[
    list[str], list[str], list[list[str, str, str]]]:
    """ Merges all KGs except the excluded KG, if fine_tuned is True the training KG is not excluded.

    :param str fine_tuned_on: Dataset on which the vocabulary is fine-tuned on, empty string results in no fine-tuning,
                              defaults to ""
    :param dict included_KGs_dict: Keys form a list of Datasets which are mapped to the names of the KGs for vocabulary building,
                                   Any KG in the dataset fine_tuned_on will be limited to train.txt only.
    :return: tuple (entities, relation_names, edges)
        WHERE
        list entities: list containing string representations of entities
        list relation_names: list containing string representations of relation_names
        list edges: list containing triples of entities and relation names representing edges
    """

    kg_dataframe = pd.DataFrame(columns=['entity1', 'rel', 'entity2'], dtype='str')
    # For Graphs from the finetuned DS, only add train.txt
    if fine_tuned_on:
        for kg in included_KGs_dict[fine_tuned_on]:
            with open(os.path.join(fine_tuned_on, kg, "train.txt")) as fin:
                df = pd.read_csv(fin, sep='\t', header=None, names=['entity1', 'rel', 'entity2'], dtype=str)
            kg_dataframe = pd.concat([kg_dataframe, df])
    # For all other Graphs add all .txt
    for ds in [ds for ds in included_KGs_dict.keys() if ds != fine_tuned_on]:
        for kg in included_KGs_dict[ds]:
            for file in os.listdir(os.path.join(ds, kg)):
                if any([file.endswith(txt) for txt in ["train.txt", "valid.txt", "test.txt"]]):
                    with open(os.path.join(ds, kg, file)) as fin:
                        df = pd.read_csv(fin, sep='\t', header=None, names=['entity1', 'rel', 'entity2'], dtype=str)
                    kg_dataframe = pd.concat([kg_dataframe, df])

    # Combines previous graphs into one KG without duplicates
    kg_dataframe = kg_dataframe.drop_duplicates()
    entities = list(set(kg_dataframe["entity1"].unique()).union(set(kg_dataframe["entity2"].unique())))
    relation_names = list(kg_dataframe["rel"].unique())
    edges = kg_dataframe.values.tolist()
    return entities, relation_names, edges


def random_walks(entities: list[str], relation_names: list[str], edges: list[list[str, str, str]], max_depth: int = 10,
                 num_walks: int = 10, random_state: int = 42) -> list[list[str]]:
    """
    :param list entities: list of entities
    :param list relation_names: list of relation_names
    :param list edges: list of edges
    :param int max_depth: maximal length of a walk, defaults to 10
    :param int num_walks: number of random walks, defaults to 5
    :param int random_state: random seed, defaults to 42

    :return: list of random walks in the form of lists of strings
    """
    random.seed(random_state)
    walks = []
    edge_df = pandas.DataFrame(edges, columns=['entity1', 'rel', 'entity2'])
    counter = 0
    # Precomp: Searches all related edges of each node
    head_to_edge_dict = {}
    for entity in entities:
        connected_edges = (edge_df.loc[edge_df["entity1"] == entity]).values.tolist()
        head_to_edge_dict.update({entity: [item[1:] for item in connected_edges]})
    # Precomp: Searches the nodes that come after each relation
    rel_to_tail_dict = {}
    for rel in relation_names:
        connected_nodes = (edge_df.loc[edge_df["rel"] == rel]).get("entity2").unique().tolist()
        rel_to_tail_dict.update({rel: connected_nodes})
    # Generates num_walks amount of walks for each node as a head
    for entity in list(entities):
        for walk_number in range(num_walks):
            walk = [entity]
            for step in range(max_depth):
                if head_to_edge_dict.get(walk[-1], False):
                    walk.extend(random.choice(head_to_edge_dict.get(walk[-1], [])))
                else:
                    break
            walks.append(walk)
    # Generates num_walks amount of walks for each relation name
    for rel in list(relation_names):
        if counter % ((len(entities) + len(relation_names)) // 10) == 0:
            print(counter // ((len(entities) + len(relation_names)) // 10))
        counter += 1
        for walk_number in range(num_walks):
            walk = [rel, random.choice(rel_to_tail_dict.get(rel))]
            for step in range(max_depth - 1):
                if head_to_edge_dict.get(walk[-1], False):
                    walk.extend(random.choice(head_to_edge_dict.get(walk[-1], [])))
                else:
                    break
            walks.append(walk)
    return walks


def corpus_creation(file_path: str = "KGs", training_KG: str = "NL-0", pretrain_KGs: list[str] = [],
                    KG_pool="KG-custom", corpus_input="VandR") -> list[str]:
    """ Creates a corpus from a training_KG. Programmed for directory structure: KGs/KG/subKG

    :param str file_path: Path of all accessible KG
    :param str training_KG: str included in the used training KG.
    :param KG_pool: The KG from which the corpus is extracted, defaults to "KG-custom".
           Type: str, {"KG-custom", "KG-pretrained", "KG-finetuned"}
    :type KG_pool: str, {"KG-custom", "KG-pretrained", "KG-finetuned"}
    :param corpus_input: The part of the KGs from which the corpus is extracted, defaults to "VandR".
           Type: str, {"VandR", "E", "randomWalks"}
    :type corpus_input: str, {"VandR", "E", "randomWalks"}

    :return: A list of strings including all entities and relation names.
    """
    included_KGs_dict = defaultdict(list)
    for root, dirs, files in os.walk(file_path):
        if "train.txt" in files:
            if os.path.basename(root) in [training_KG] + pretrain_KGs:
                if os.path.basename(root) == training_KG:
                    training_ds = os.path.dirname(root)
                included_KGs_dict[os.path.dirname(root)].append(os.path.basename(root))
    included_count = len([value for values in included_KGs_dict.values() for value in values])
    assert (included_count == len(pretrain_KGs) + 1), (included_count, len(pretrain_KGs))
    if "KG-custom" == KG_pool:
        included_KGs_dict = {training_ds: included_KGs_dict[training_ds]}
        entities, relation_names, edges = KG_selection(fine_tuned_on=training_ds, included_KGs_dict=included_KGs_dict)
    elif "KG-pretrained" == KG_pool:
        entities, relation_names, edges = KG_selection(fine_tuned_on="", included_KGs_dict=included_KGs_dict)
    elif "KG-finetuned" == KG_pool:
        entities, relation_names, edges = KG_selection(fine_tuned_on=training_ds, included_KGs_dict=included_KGs_dict)
    match corpus_input:
        case "VandR":
            return [*entities, *relation_names]
        case "E":
            return [string for edge in edges for string in edge]
        case "randomWalks":
            walks = random_walks(entities=entities, relation_names=relation_names, edges=edges, max_depth=5,
                                 num_walks=1)
            return [string for walk in walks for string in walk]
        case _:
            raise ValueError('Not one of the corpus input types {"VandR", "E", "randomWalks"}')


def remove_white_spaces(data: list[list[str]]) -> list[list[str]]:
    result = []
    for line in data:
        subres = []
        for string in line:
            subres.append(string.replace(" ", "_"))
        result.append(subres)
    return result
