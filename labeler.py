import csv
import os
import time

import requests
from unidecode import unidecode


def wikidata_ids_to_labels(ids: list[str]):
    '''
    
    :param ids: 
    :return: 
    '''

    query = '''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        
        SELECT ?element ?label
        WHERE {
          VALUES ?element {%s }
          ?element rdfs:label ?label .
          FILTER (lang(?label) = "en")
        }
        '''
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    result = {}
    for id_group in [ids[x:x + 500] for x in range(0, len(ids), 500)]:
        values = " wd:".join([''] + [group for group in id_group])
        print(len(id_group))
        try:
            temp = requests.get(url=url, params={'query': query % values, 'format': 'json'})
            data = temp.json()
            time.sleep(1)
        except requests.exceptions.RequestException:
            print(values)
            print(temp)
            break
        for element in data['results']['bindings']:
            wd = element['element']['value'].replace('http://www.wikidata.org/entity/', '')
            label = unidecode(element['label']['value'])
            result.update({wd: label})
    return result


def entity_labels_from_wikidata(wikidata_df, entities):
    """ Replaces Freebase IDs with textual labels; does not translate unknown IDs

    :param wikidata_df: DataFrame containing wikidata of entities
    :param entities: List of entities to be replaced
    :return: list of textual representations of entities
    """
    entity_dict = wikidata_df.transpose()["label"].to_dict()
    return [entity_dict.get(wdid, wdid) for wdid in entities]


def fbk_dataset_to_textual(dataset, entity_dict):
    """ Replaces Freebase IDs with textual labels; does not translate unknown IDs

    :param list dataset: list of edges
    :param dict entity_dict: dict mapping entity IDs to textual labels
    :return: list of edges with textual entity labels
    """
    res = dataset
    for line in range(len(dataset)):
        for string in range(len(dataset[line])):
            entity = dataset[line][string]
            # Create an ASCII representation, non ASCII letters are transliterated
            res[line][string] = unidecode(entity_dict.get(entity, entity))
    return res

def wikidata_dataset_to_textual(dataset, entity_dict):
    """ Replaces wikidata IDs with textual labels; does not translate unknown IDs

    :param list dataset: list of edges
    :param dict entity_dict: dict mapping entity IDs to textual labels
    :return: list of edges with textual entity labels
    """
    res = dataset
    for line in range(len(dataset)):
        for string in range(len(dataset[line])):
            entity = dataset[line][string]
            # Create an ASCII representation, non ASCII letters are transliterated
            res[line][string] = entity_dict.get(entity, entity)
    return res

def edge_dict_from_wn18RR(non_text_file_path, text_file_path):
    """ replaces wordnet offsets with textual representations.
    textual representations from: https://github.com/villmow/datasets_knowledge_embedding

    :param str non_text_file_path: path to non-textual datasets
    :param str text_file_path: path to textual datasets
    :return: dictionary mapping edges to textual representations
    """
    result = {}
    files = os.listdir(non_text_file_path)
    for file in files:
        print(os.path.join(non_text_file_path, file))
        print(os.path.join(text_file_path, file))
        dataset1 = list(csv.reader(open(os.path.join(non_text_file_path, file)), delimiter="\t"))
        dataset2 = list(csv.reader(open(os.path.join(text_file_path, file)), delimiter="\t"))
        dataset1 = [", ".join(edge) for edge in dataset1]
        result.update(zip(dataset1, dataset2))
    return result


def wn18_dataset_to_textual(dataset, edge_dict):
    """ Replaces WWN18RR offsets with textual labels

    :param list dataset: list of edges
    :param dict edge_dict: dict mapping comma-seperated edges to textual labels
    :return: list of edges with textual entity labels
    """
    res = dataset
    for line in range(len(dataset)):
        res[line] = edge_dict.get(", ".join(dataset[line]), dataset[line])
        for string in range(len(dataset[line])):
            entity = dataset[line][string]
            # Create an ASCII representation, non ASCII letters are transliterated
            res[line][string] = unidecode(edge_dict.get(entity, entity))
    return res
