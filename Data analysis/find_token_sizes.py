import argparse
import csv
import os
import pickle
from collections import OrderedDict

import tiktoken

from custom_dice_executer import CustomExecute
from data_preperation import corpus_creation
from dicee import Execute
from dicee.config import Namespace

dicee_args = Namespace()
dicee_args.byte_pair_encoding = True
dicee_args.trainer = "PL"
dicee_args.scoring_technique = "KvsAll"
dicee_args.eval_model = "train_val_test"
dicee_args.path_to_store_single_run = "Experiments/Whatever"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DS", type=str, default="")
    args = parser.parse_args()


    def csv_entry(dataset, max_train_size, max_valid_size, max_test_size, vocab="default", vocab_size=""):
        result = OrderedDict()
        result["dataset"] = os.path.basename(root)
        result["vocab"] = vocab
        result["vocab_size"] = vocab_size
        result["max_train_size"] = max_train_size
        result["max_valid_size"] = max_valid_size
        result["max_test_size"] = max_test_size
        result["truncation needed"] = max_train_size < max_valid_size or max_train_size < max_valid_size
        return result


    csv_list = []
    for root, dirs, files in os.walk("KGs/" + args.DS):
        if "train.txt" in files:
            print(root)
            dicee_args.dataset_dir = root
            executer = Execute(dicee_args)
            loaded_kg = executer.read_or_load_kg()
            csv_list.append(csv_entry(os.path.basename(root), loaded_kg.max_length_train_tokens,
                                      loaded_kg.max_length_val_tokens, loaded_kg.max_length_test_tokens))
            if os.path.isdir(root.replace("KGs", "Vocabularies")):
                element_count = len(corpus_creation(file_path="KGs", training_KG=os.path.basename(root),
                                                    KG_pool="KG-custom", corpus_input="VandR"))
                voc_path = root.replace("KGs", "Vocabularies")
                vocabs = []
                for voc_root, voc_dirs, voc_files in os.walk(voc_path):
                    for voc_file in voc_files:
                        if "_vocab.pkl" in voc_file:
                            vocabs.append(os.path.join(voc_root, voc_file))
                print(vocabs)
                for vocab in vocabs:
                    for size in [0.5, 1, 2]:
                        with open(vocab, "rb") as f:
                            initial_ranks = pickle.load(f)
                            ranks = dict(sorted(initial_ranks.items(), key=lambda x: x[1])[
                                         :min([int(size * element_count), len(initial_ranks)])])
                        enc = tiktoken.Encoding(name="Custom_tokenizer", mergeable_ranks=ranks,
                                                pat_str=r"[\s:_]|[^\s:_]+",
                                                special_tokens={})
                        executer = CustomExecute(dicee_args, vocabulary=enc)
                        loaded_kg = executer.read_or_load_kg()
                        csv_list.append(
                            csv_entry(os.path.basename(root), loaded_kg.max_length_train_tokens,
                                      loaded_kg.max_length_val_tokens, loaded_kg.max_length_test_tokens, vocab=vocab,
                                      vocab_size=size))
    keys = csv_list[0].keys()
    os.makedirs("Data analysis/Useful_stats", exist_ok=True)
    with open('Data analysis/Useful_stats/%s_token_sizes.csv'%args.DS, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_list)
