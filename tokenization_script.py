import os
import pickle
import argparse
import json

import custom_BytE_util
from custom_tokenizer import *
from data_preperation import corpus_creation

parser = custom_BytE_util.CustomBytEParser()
parser.add_argument("--pretrain_KGs", nargs="*")
parser.add_argument("--logging", help="Statistics about the vocabulary are logged.", action="store_true")
parser.add_argument("--overwrite", help="Overwrites previous vocabularies.", action="store_true")
args = parser.parse_args()
print(args)

statistics_dict = {"Training-KG": args.training_KG}
vocab_path = ""
for root, dirs, files in os.walk(args.KG_path):
    if args.training_KG in dirs:
        if args.KG_pool == "KG-pretrained":
            vocab_path = root.replace(args.KG_path, os.path.join("Vocabularies", ""))
        else:
            vocab_path = os.path.join(root.replace(args.KG_path, os.path.join("Vocabularies", "")), args.training_KG)
        vocab_path = os.path.join(vocab_path, args.KG_pool, args.corpus_input)
if not vocab_path:
    print("KG not found.")
data = corpus_creation(file_path=args.KG_path, training_KG=args.training_KG, pretrain_KGs=args.pretrain_KGs,
                       KG_pool="KG-custom", corpus_input="VandR")
statistics_dict["|V| + |R|"] = element_count = len(data)
timer_start = time.time()
data = corpus_creation(file_path=args.KG_path, training_KG=args.training_KG, pretrain_KGs=args.pretrain_KGs,
                       KG_pool=args.KG_pool, corpus_input=args.corpus_input)
statistics_dict["Corpus creation time"] = time.time() - timer_start
statistics_dict["Corpus size"] = len(data)

os.makedirs(vocab_path, exist_ok=True)
with open(os.path.join(vocab_path, "corpus.pkl"), 'wb') as f:
    pickle.dump(data, f)

vocab_file_path = os.path.join(vocab_path, "_".join([args.tie_breaker, "vocab.pkl"]))
statistics_file_path = os.path.join(vocab_path, "_".join([args.tie_breaker, "statistics.json"]))
if not os.path.isfile(vocab_file_path) or args.overwrite:
    start_time = time.time()
    print("Attempting to create %s." % vocab_file_path)
    ranks = faster_bpe(data=data, vocab_size=min(max(50000, 3 * element_count), 100000),
                       tie_breaker=args.tie_breaker)
    statistics_dict["Vocabulary building time"] = time.time() - start_time

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_file_path, "wb") as f:
        pickle.dump(ranks, f)
    with open(statistics_file_path, "w") as f:
        json.dump(statistics_dict, f, indent=2)
elif not args.overwrite:
    print("%s already exists." % vocab_file_path)
