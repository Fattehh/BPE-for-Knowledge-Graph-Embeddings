import os
import sys
from itertools import chain

import pandas as pd

sys.path.insert(1, "Data analysis")

import pickle

# noinspection PyUnresolvedReferences
from table_generator import create_table
# noinspection PyUnresolvedReferences
from select_best_parameters import best_parameters_from_path, select_best_parameters
from custom_BytE_util import Benchmarker, ParameterSpace


def get_summary(name, tabulate: bool, params: list[list[str]]) -> (pd.DataFrame, list[ParameterSpace]):
    all_sums_path = os.path.join("automated_scripts", "Summaries")
    sum_path = os.path.join("automated_scripts", name)
    os.makedirs(all_sums_path, exist_ok=True)
    os.makedirs(os.path.join(sum_path, "Summaries"), exist_ok=True)
    if tabulate:
        # Get the parameterspaces with the best vocabularies
        sum_df = create_table(sum_path)
        with open(os.path.join(all_sums_path, "%s_sum_df.pkl" % name), "wb") as f:
            pickle.dump(sum_df, f)
        best_params = [best_parameters_from_path(sum_path, param, gpt=False) for param in params]
    else:
        with open(os.path.join(all_sums_path, "%s_sum_df.pkl" % name), "rb") as f:
            sum_df = pickle.load(f)
        best_params = []
        for file in [os.path.join(sum_path, "Summaries", summ + "Best_parameters.pkl") for summ in
                     ["_".join(p) + "_" if p else "" for p in params]]:
            with open(file, "rb") as f:
                best_params.append(pickle.load(f))
    return sum_df, best_params


custom_parameters = {"training_KG": "nell_v2_ind", "KG_pool": "KG-custom", "corpus_input": "VandR",
                     "tie_breaker": "ascending_size", "vocab_size": 4.0, "model": "ComplEx", "embedding_dim": 32,
                     "lr": 0.01, "num_epochs": 1000, "min_epochs": 500, "base_seed": 0, "forced_truncation": 100,
                     "bpe_truncation": "None", "bpe_with_RNN": "Linear", "multiple_bpe_encodings": "1"}
orig_parameters = {"training_KG": "nell_v2_ind", "KG_pool": "original_BytE", "model": "ComplEx", "embedding_dim": 32,
                   "lr": 0.01, "num_epochs": 1000, "min_epochs": 500, "base_seed": 0, "forced_truncation": 100,
                   "bpe_truncation": "None", "bpe_with_RNN": "Linear", "multiple_bpe_encodings": "1"}
ds_dict = {'DB100K_all': ['DB100K_labeled'],
           'dice': ["NELL-995-h25", "NELL-995-h50", "NELL-995-h75", "NELL-995-h100"],
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

new_vocabularies = True
training_kgs = ["FB-v3-ind", "NL-v1-ind", "NL-v4-ind", "wn-v2-ind", "wn-v4-ind", "NL-25", "NELL-995-h75"]
if new_vocabularies:
    ind_train_kgs = ["fb237_v3_ind_uslabeled", "nell_v1_ind", "nell_v4_ind", "WN18RR_v2_ind_labeled",
                     "WN18RR_v4_ind_labeled"]
    pretrain_kgs = []
    for kg in training_kgs:
        pretrain_kg = [value for values in ds_dict.values() for value in values if
                       (kg not in values and "NELL-995-h75" not in values and not all(
                           ["NELL-995" in x for x in [kg, values]]))]
        if ind_train_kgs:
            pretrain_kg = [ind_train_kgs.pop(0)] + pretrain_kg
        pretrain_kgs.append(pretrain_kg)
    Benchmarker.create_tokenization_scripts("new_vocabularies", training_kgs, pretrain_kgs)

explore_vocabs = False
vocabs_bench = True
explore_linears = True
linears_bench = True
explore_truncs = True
truncs_bench = True
explore_multis = True
multis_bench = True
tabulate_vocabs = True
tabulate_linears = True
tabulate_truncs = True
tabulate_multis = True

paramspaces = []
model_modes = {"model": ["ComplEx", "Keci"], "embedding_dim": ["16", "32", "64"], "lr": ["0.1", "0.01"]}
if explore_vocabs:
    paramspaces.append(ParameterSpace(
        {"KG_path": "KGs", "training_KG": training_kgs[0], "KG_pool": "KG-custom", "corpus_input": "VandR",
         "tie_breaker": "ascending_size", "vocab_size": 4.0, "model": "ComplEx", "embedding_dim": 32, "lr": 0.01,
         "num_epochs": 100, "min_epochs": 500, "base_seed": 0, "forced_truncation": 100, "bpe_truncation": "None",
         "bpe_with_RNN": "Linear", "multiple_bpe_encodings": "1", "fix_missing": 0}))
    sep_modes = [
        {"KG_pool": ["KG-custom"], "corpus_input": ["explore"], "tie_breaker": ["ascending_size", "descending_size"],
         "vocab_size": ["0.5", "1.0", "2.0"], },
        {"KG_pool": ["KG-pretrained", "KG-finetuned"], "corpus_input": ["VandR"],
         "tie_breaker": ["ascending_size", "descending_size"], "vocab_size": ["0.5", "1.0", "2.0"], },
        {"KG_pool": ["original_BytE"]}]
    modes = model_modes
    vocab_explorer = Benchmarker("vocab_exploration", paramspaces, modes, sep_modes, 1, training_KGs=training_kgs)
    vocab_explorer.create_benchmark_scripts()
vocab_sum_df, best_vocab_params = get_summary("vocab_exploration", tabulate_vocabs, [[], ["KG_pool"]])

if vocabs_bench:
    pspaces = list(chain.from_iterable(best_vocab_params[1:]))
    for pspace in pspaces:
        pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
        pspace.parameters["num_epochs"] = pspace.num_epochs = "1000"
    modes = {"fix_missing": ["4"], "base_seed": ["10"]}
    vocabs_bencher = Benchmarker("vocab_benchmarks", pspaces, modes, [], 5)
    vocabs_bencher.create_benchmark_scripts()

if explore_linears:
    modes = {"bpe_with_RNN": ["RNN", "GRU", "LSTM"], "num_epochs": ["200"], "min_epochs": ["500"], "base_seed": ["0"],
             "fix_missing": ["0"]}
    modes.update(model_modes)
    linears_explorer = Benchmarker("linear_exploration", best_vocab_params[0], modes, [], 1)
    linears_explorer.create_benchmark_scripts()
linear_sum_df, best_linear_params = get_summary("linear_exploration", tabulate_linears, [["bpe_with_RNN"]])

if linears_bench:
    pspaces = list(chain.from_iterable(best_linear_params))
    for pspace in pspaces:
        pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
        pspace.parameters["num_epochs"] = pspace.num_epochs = "2000"
    modes = {"fix_missing": ["4"], "base_seed": ["10"]}
    linears_bencher = Benchmarker("linear_benchmarks", pspaces, modes, [], 5)
    linears_bencher.create_benchmark_scripts()

best_unmodified_params = select_best_parameters("automated_scripts", "unmodified", pd.concat([vocab_sum_df, linear_sum_df]),
                                         gpt=False)

if explore_truncs:
    pspaces = best_unmodified_params
    for pspace in pspaces:
        pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
    modes = {"forced_truncation": ["90", "95"],
             "bpe_truncation": ["last_first", "asc_size", "desc_size", "asc_rank", "desc_rank"],
             "base_seed": ["0"], "fix_missing": ["0"]}
    modes.update(model_modes)
    truncs_explorer = Benchmarker("trunc_exploration", pspaces, modes, [], 1)
    truncs_explorer.create_benchmark_scripts()
trunc_sum_df, best_truncs_params = get_summary("trunc_exploration", tabulate_truncs, [["forced_truncation"],["bpe_truncation"]])

if truncs_bench:
    pspaces = list(chain.from_iterable(best_truncs_params))
    plist = []
    final_pspaces = []
    for pspace in pspaces:
        if pspace.bpe_with_RNN == "Linear":
            pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
            pspace.parameters["num_epochs"] = pspace.num_epochs = "1000"
        else:
            pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
            pspace.parameters["num_epochs"] = pspace.num_epochs = "2000"
        if pspace.parameters not in plist:
            plist.append(pspace.parameters)
            final_pspaces.append(pspace)
    modes = {"fix_missing": ["4"], "base_seed": ["10"]}
    truncs_bencher = Benchmarker("trunc_benchmarks", final_pspaces, modes, [], 5)
    truncs_bencher.create_benchmark_scripts()

if explore_multis:
    pspaces = best_unmodified_params
    for pspace in pspaces:
        pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
    modes = {"multiple_bpe_encodings": ["2", "3", "4"], "base_seed": ["0"],
             "fix_missing": ["0"]}
    modes.update(model_modes)
    multis_explorer = Benchmarker("multi_exploration", pspaces, modes, [], 1)
    multis_explorer.create_benchmark_scripts()
multis_sum_df, best_multis_params = get_summary("multi_exploration", tabulate_multis, [["multiple_bpe_encodings"]])

if multis_bench:
    pspaces = list(chain.from_iterable(best_multis_params))
    for pspace in pspaces:
        if pspace.bpe_with_RNN == "Linear":
            pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
            pspace.parameters["num_epochs"] = pspace.num_epochs = "1000"
        else:
            pspace.parameters["min_epochs"] = pspace.min_epochs = "500"
            pspace.parameters["num_epochs"] = pspace.num_epochs = "2000"
    modes = {"fix_missing": ["4"], "base_seed": ["10"], "multiple_bpe_loss": ["False", "True"]}
    multis_bencher = Benchmarker("multi_benchmarks", pspaces, modes, [], 5)
    multis_bencher.create_benchmark_scripts()
