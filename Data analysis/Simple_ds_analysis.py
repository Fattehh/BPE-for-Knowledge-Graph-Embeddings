import os
import pickle
import statistics

from Levenshtein import distance
import pandas as pd
from custom_BytE_util import dir_path

ds_dict = {'DB100K_all': ['DB100K_labeled'],
           'FB15K237_all': ['FB-100_uslabeled', 'FB-25_uslabeled', 'FB-50_uslabeled', 'FB-75_uslabeled', 'FB-v1-ind',
                            'FB-v2-ind', 'FB-v3-ind', 'FB-v4-ind', 'FB15K237_uslabeled', 'fb237_v1_ind_uslabeled',
                            'fb237_v1_uslabeled', 'fb237_v2_ind_uslabeled', 'fb237_v2_uslabeled',
                            'fb237_v3_ind_uslabeled', 'fb237_v3_uslabeled', 'fb237_v4_ind_uslabeled',
                            'fb237_v4_uslabeled'],
           'NELL-995_all': ['NELL-995', 'NELL-995-v1', 'nell_v1', 'nell_v1_ind', 'nell_v2', 'nell_v2_ind', 'nell_v3',
                            'nell_v3_ind', 'nell_v4', 'nell_v4_ind', 'NL-0', 'NL-100', 'NL-25', 'NL-50', 'NL-75',
                            'NL-v1-ind', 'NL-v2-ind', 'NL-v3-ind', 'NL-v4-ind'],
           'Wikidata': ['WK-100_uslabeled', 'WK-25_uslabeled', 'WK-50_uslabeled', 'WK-75_uslabeled'],
           'WNN18RR_all': ['wn-v1-ind', 'wn-v2-ind', 'wn-v3-ind', 'wn-v4-ind', 'WN18RR_labeled',
                           'WN18RR_v1_ind_labeled', 'WN18RR_v1_labeled', 'WN18RR_v2_ind_labeled', 'WN18RR_v2_labeled',
                           'WN18RR_v3_ind_labeled', 'WN18RR_v3_labeled', 'WN18RR_v4_ind_labeled', 'WN18RR_v4_labeled']}
training_kgs = []
pretrain_kgs = []

for kg in [("FB", "fb237", "_ind_uslabeled"), ("NL", "nell", "_ind"), ("wn", "WN18RR", "_ind_labeled")]:
    for i in [1, 2, 3, 4]:
        tr_kg = kg[0] + "-v%d-ind" % i
        training_kgs.append(tr_kg)
        pretrain_kg = [value for values in ds_dict.values() for value in values if tr_kg not in values]
        pretrain_kgs.append([kg[1] + "_v%d" % i + kg[2]] + pretrain_kg)
training_kgs.extend(
    ['NL-0', 'NL-100', 'NL-25', 'NL-50', 'NL-75', "NELL-995-h25", "NELL-995-h50", "NELL-995-h75", "NELL-995-h100"])
# training_kgs.extend(["NELL-995-h25-ind", "NELL-995-h50-ind", "NELL-995-h75-ind", "NELL-995-h100-ind"])


def single_ds_analysis(ds_path: str) -> (int, int, int, int, int, int, int, int, float):
    output = []
    train_df = pd.read_csv(os.path.join(ds_path, "train.txt"), sep='\t', header=None,
                           names=['entity1', 'rel', 'entity2'], dtype=str)
    test_df = pd.read_csv(os.path.join(ds_path, "test.txt"), sep='\t', header=None,
                          names=['entity1', 'rel', 'entity2'], dtype=str)
    train_entities = set(train_df["entity1"].unique()) | set(train_df["entity2"].unique())
    train_relations = set(train_df["rel"].unique())
    train_num_edges = len(train_df.index)
    test_entities = set(test_df["entity1"].unique()) | set(test_df["entity2"].unique())
    test_relations = set(test_df["rel"].unique())
    test_num_edges = len(test_df.index)
    unknown_entities = test_entities - train_entities
    unknown_relations = test_relations - train_relations
    mean_min_distance = []
    for entity in test_entities:
        mean_min_distance.append(min([distance(entity, train_entity) for train_entity in train_entities]))
    for relation in test_relations:
        mean_min_distance.append(min([distance(relation, train_relation) for train_relation in train_relations]))
    mean_min_distance = statistics.fmean(mean_min_distance)
    return len(train_entities), len(train_relations), train_num_edges, len(test_entities), len(
        test_relations), test_num_edges, len(unknown_entities), len(unknown_relations), mean_min_distance


analyses = []
new_analysis = False
if new_analysis:
    for kg in training_kgs:
        for root, dirs, files in os.walk("KGs"):
            if kg in root:
                print(kg)
                analysis = (kg,) + single_ds_analysis(os.path.join(root))
                analyses += [analysis]
                break
    file = open("Data analysis/Useful_stats/ds_stats.pkl", "wb")
    pickle.dump(analyses, file)
else:
    file = open("Data analysis/Useful_stats/ds_stats.pkl", "rb")
    analyses = pickle.load(file)
df = pd.DataFrame(analyses,
                  columns=["KG", ("train", "Entities"), ("train", "Relations"), ("train", "triples"), ("test", "Entities"),
                           ("test", "Relations"), ("test", "triples"), ("test", "unknown_entities"),
                           ("test", "unknown_relations"), ("test", "mean_levendistance")])
print(df.to_string())
