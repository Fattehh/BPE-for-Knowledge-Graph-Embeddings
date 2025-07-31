import collections
import csv
import statistics

import numpy as np
import pandas as pd

from custom_tokenizer import *
from labeler import *
from data_preperation import *
import pickle


def labeled_FB():
    """ With some exceptions converts all FB IDs in FB15k237 to rdf labels."""
    # First load json containing wikidata of FB15k237
    wikidata_of_fb15k = pd.read_json("Extra data/FB15k entity2wikidata.json")
    # Select all entities from datasets created from FB15k237
    entity_dict = wikidata_of_fb15k.transpose()["label"].to_dict()

    for root, dirs, files in os.walk("KGs/FB15K237_all"):
        if "_labeled".casefold() not in root.casefold():
            # Generates labeled datasets for each unlabeled one
            for file in set(files).intersection(["test.txt", "train.txt", "valid.txt"]):
                current_file_path = os.path.join(root, file)
                textual_file_path = os.path.join(root + "_labeled", file)
                dataset = list(csv.reader(open(current_file_path), delimiter="\t"))
                text_dataset = fbk_dataset_to_textual(dataset=dataset, entity_dict=entity_dict)
                os.makedirs(os.path.dirname(textual_file_path), exist_ok=True)
                with open(textual_file_path, 'w', newline="", encoding="ascii") as csfile:
                    csv.writer(csfile, delimiter="\t").writerows(text_dataset)


def labeled_WN18RR():
    """ Replaces non-textual edges with textual ones"""
    # Get the mapping for textual edges for each non-textual edge in the original dataset
    edge_dict = edge_dict_from_wn18RR("KGs/WNN18RR_all/WN18RR", "Extra data/WN18RR_textual")
    for root, dirs, files in os.walk("KGs/WNN18RR_all"):
        if "_labeled".casefold() not in root.casefold():
            # Use the previous mapping to generate textual datasets
            for file in set(files).intersection(["test.txt", "train.txt", "valid.txt"]):
                current_file_path = os.path.join(root, file)
                textual_file_path = os.path.join(root + "_labeled", file)
                dataset = list(csv.reader(open(current_file_path), delimiter="\t"))
                text_dataset = wn18_dataset_to_textual(dataset=dataset, edge_dict=edge_dict)
                os.makedirs(os.path.dirname(textual_file_path), exist_ok=True)
                with open(textual_file_path, 'w', newline="", encoding="ascii") as csfile:
                    csv.writer(csfile, delimiter="\t").writerows(text_dataset)

# deprecated
def build_wikidata_dict():
    """ Builds dictionary for wikidata, Can fail to find labels
        I assume entities with default labels do not work for some reason"""
    entities, relation_names, edges = KG_selection(file_path="KGs", fine_tuned_on="",
                                                   included_KGs=["db100"])
    wikidata_elements = [*entities, *relation_names]
    wikidata_elements = [word for word in wikidata_elements if
                         word[0] in ['Q', 'P'] and all([letter.isdigit() for letter in word[1:]])]
    with open("Extra data/wikidata_dict.txt", "r") as file:
        prev_dict = dict(csv.reader(file, delimiter="\t"))
    leftovers = list(set(wikidata_elements) - set(prev_dict.keys()))
    print(leftovers)
    while len(leftovers) > 0:
        with open("Extra data/wikidata_dict.txt", "r") as file:
            prev_dict = dict(csv.reader(file, delimiter="\t"))
        leftovers = list(set(wikidata_elements) - set(prev_dict.keys()))
        print("hellooooo", len(leftovers))
        next_dict = wikidata_ids_to_labels(leftovers)
        next_dict.update(prev_dict)
        with open("Extra data/wikidata_dict.txt", 'w', newline="", encoding="ascii") as csfile:
            csv.writer(csfile, delimiter="\t").writerows(next_dict.items())


def labeled_wikidata():
    """ Replaces non-textual edges with textual ones"""
    # Load entity dict
    with open("Extra data/wikidata_dict.txt", "r") as file:
        entity_dict = dict(csv.reader(file, delimiter="\t"))
    for root, dirs, files in os.walk("KGs/wikidata"):
        if "_labeled".casefold() not in root.casefold():
            # Generates labeled datasets for each unlabeled one
            for file in set(files).intersection(["test.txt", "train.txt", "valid.txt"]):
                current_file_path = os.path.join(root, file)
                textual_file_path = os.path.join(root + "_labeled", file)
                dataset = list(csv.reader(open(current_file_path), delimiter="\t"))
                text_dataset = wikidata_dataset_to_textual(dataset=dataset, entity_dict=entity_dict)
                os.makedirs(os.path.dirname(textual_file_path), exist_ok=True)
                with open(textual_file_path, 'w', newline="", encoding="ascii") as csfile:
                    csv.writer(csfile, delimiter="\t").writerows(text_dataset)


def labeled_dbpedia():
    """ Replaces non-textual edges with textual ones
    Fails to replace around 300 entities. Those seem to be redirected entities on wikidata"""
    # Load entity dict
    with open("Extra data/wikidata_dict.txt", "r") as file:
        entity_dict = dict(csv.reader(file, delimiter="\t"))
    for root, dirs, files in os.walk("KGs/DB100K_all"):
        if "_labeled".casefold() not in root.casefold():
            print("Hello")
            # Generates labeled datasets for each unlabeled one
            for file in set(files).intersection(["test.txt", "train.txt", "valid.txt"]):
                current_file_path = os.path.join(root, file)
                textual_file_path = os.path.join(root + "_labeled", file)
                dataset = list(csv.reader(open(current_file_path), delimiter="\t"))
                text_dataset = wikidata_dataset_to_textual(dataset=dataset, entity_dict=entity_dict)
                os.makedirs(os.path.dirname(textual_file_path), exist_ok=True)
                with open(textual_file_path, 'w', newline="", encoding="ascii") as csfile:
                    csv.writer(csfile, delimiter="\t").writerows(text_dataset)


# data = corpus_creation(file_path="KGs/", training_KG="wk-100", KG_pool="KG-Custom", corpus_input="VandR")[:2000]
# element_count = len(data)
# print(element_count)
# ranks = FasterBPE(data=data,vocab_size=max(50000, 2 * element_count), tie_breaker="default")
# print(ranks)
# ranks = kge_bpe_train(data=data,vocab_size=max(50000, 2 * element_count), tie_breaker="default")
# print(ranks)

# with open("KGs/FB15K237_all/FB-100_labeled/.pkl", "rb") as f:
#     vocab = pickle.load(f)
# print(len(vocab))
# before = time.time()
# with open("KGs/Wikidata/WK-25_labeled/train.txt", 'r', encoding="ascii") as f:
#     csv_text = list(csv.reader(f, delimiter="\t"))
for root, dirs, files in os.walk("KGs/FB15K237_all"):
    for file in files:
        with open(os.path.join(root,file), 'r', encoding="ascii") as f:
            csv_text = list(csv.reader(f, delimiter="\t"))
        us_path = os.path.join(root,file).replace("_labeled", "_uslabeled")
        us_text = remove_white_spaces(csv_text)
        os.makedirs(root.replace("_labeled", "_uslabeled"), exist_ok=True)
        print(us_text)
        with open(us_path, 'w', newline="", encoding="ascii") as csfile:
            csv.writer(csfile, delimiter="\t").writerows(us_text)

# data = corpus_creation(file_path="KGs/", training_KG="wk-100", KG_pool="KG-Custom", corpus_input="VandR")
# print(data)
# element_count = len(data)
# print(element_count)
# ranks = kge_bpe_train(data=data,vocab_size=1000, tie_breaker="default")
# with open("Extra data/wikidata_vocab", "wb") as f:
#     pickle.dump(ranks, f)
# with open("Extra data/wikidata_vocab", "rb") as f:
#     ranks = pickle.load(f)
#
# print(type(list(ranks.keys())[0]))
# print(dict(sorted(ranks.items(),key=lambda x:x[1])[:300]))
# print(len(dict(sorted(ranks.items(),key=lambda x:x[1])[:300])))
# enc = tiktoken.Encoding(name="VandR_default_vocab",mergeable_ranks=ranks,pat_str=r".*",special_tokens={"<|endoftext|>": 50256})
# print(enc.decode_bytes([32]))
# print([enc.decode([32])])
# print(enc.encode(" "))
# print(enc.encode('"France"  2'), enc.decode_tokens_bytes(enc.encode('"France"  2')))

# print(np.load("Experiments/Wikidata/WK-25_labeled/KG-custom/VandR/ascending_size_vocab/0.5/ComplEx/32/0.01/Experiment000/train_set.npy"))
# entity_encodings = pd.read_pickle("Experiments/Wikidata/WK-25_uslabeled/KG-custom/VandR/default_vocab/0.5/ComplEx/32/0.01/Experiment000/ordered_bpe_entities.p")
# print((entity_encodings[0]))
# ents, lens, paddeds = zip(*entity_encodings)
# diffs = [len(tup[2]) - len(tup[1]) for tup in entity_encodings]
# print(statistics.fmean([len(ent) for ent in ents]))
# print(statistics.fmean([len(ent.encode("ascii")) for ent in ents]))
# print(max(lens,key=len))
# print(statistics.fmean(diffs))



# bad_train_df = pd.read_csv("OldKGs/wikidata/WK-25_labeled/train.txt",
#                            sep=r"\s+", header=None, usecols=[0, 1, 2], names=['subject', 'relation', 'object'],
#                            dtype=str)
# bad_valid_df = pd.read_csv("OldKGs/wikidata/WK-25_labeled/valid.txt",
#                            sep=r"\s+", header=None, usecols=[0, 1, 2], names=['subject', 'relation', 'object'],
#                            dtype=str)
# bad_test_df = pd.read_csv("OldKGs/wikidata/WK-25_labeled/test.txt",
#                           sep=r"\s+", header=None, usecols=[0, 1, 2], names=['subject', 'relation', 'object'],
#                           dtype=str)
# bad_total_df = pd.concat([bad_train_df, bad_test_df], ignore_index=True)
# train_df = pd.read_csv("KGs/Wikidata/WK-25_uslabeled/train.txt",
#                        sep=r"\s+", header=None, usecols=[0, 1, 2], names=['subject', 'relation', 'object'], dtype=str)
# test_df = pd.read_csv("KGs/Wikidata/WK-25_uslabeled/test.txt",
#                       sep=r"\s+", header=None, usecols=[0, 1, 2], names=['subject', 'relation', 'object'], dtype=str)
#
# total_df = pd.concat([train_df, test_df], ignore_index=True)
# bad_subs = set(bad_train_df["subject"]) & set(bad_test_df["subject"])
# subs = set(train_df["subject"]) & set(test_df["subject"])
# bad_known_rels = set(bad_train_df["relation"]) & set(bad_test_df["relation"])
# known_rels = set(train_df["relation"]) & set(test_df["relation"])
# bad_objs = set(bad_train_df["object"]) & set(bad_test_df["object"])
# objs = set(train_df["object"]) & set(test_df["object"])
# bad_known_ents = (set(bad_train_df["subject"]) | set(bad_train_df["object"])) & (
#         set(bad_test_df["subject"]) | set(bad_test_df["object"]))
# known_ents = (set(train_df["subject"]) | set(train_df["object"])) & (
#         set(test_df["subject"]) | set(test_df["object"]))
# bad_all_ents = (set(bad_total_df["subject"]) | set(bad_total_df["object"]))
# all_ents = (set(total_df["subject"]) | set(total_df["object"]))
# bad_all_rels = set(bad_total_df["relation"])
# all_rels = set(total_df["relation"])
# false_ents = bad_all_ents - all_ents
#
# print("Wrong read vs right read")
# print("Number of unique entities: %d vs %d" % (len(bad_all_ents), len(all_ents)))
# print("Number of unique relations: %d vs %d" % (len(bad_all_rels), len(all_rels)))
# print("Number of known entities: %d vs %d" % (len(bad_known_ents), len(known_ents)))
# print("Number of known relations: %d vs %d\n" % (len(bad_known_rels), len(known_rels)))
#
# bad_train_ents = (set(bad_train_df["subject"]) | set(bad_train_df["object"]))
# bad_train_rels = set(bad_train_df["relation"])
# new_bad_valid_df = pd.concat(
#     [bad_valid_df.loc[(bad_valid_df["subject"].isin(bad_train_ents)) | (bad_valid_df["object"].isin(bad_train_ents))],
#      bad_test_df.loc[(bad_test_df["subject"].isin(bad_train_ents)) | (bad_test_df["object"].isin(bad_train_ents))]],
#     ignore_index=True)
# new_bad_test_df = pd.concat(
#     [bad_valid_df.loc[(~bad_valid_df["subject"].isin(bad_train_ents)) & (~bad_valid_df["object"].isin(bad_train_ents))],
#      bad_test_df.loc[(~bad_test_df["subject"].isin(bad_train_ents)) & (~bad_test_df["object"].isin(bad_train_ents))]],
#     ignore_index=True)
#
# print(new_bad_valid_df.shape[0])
# print(new_bad_test_df.shape[0])
# print(pd.concat([bad_valid_df,bad_test_df]).shape[0])

