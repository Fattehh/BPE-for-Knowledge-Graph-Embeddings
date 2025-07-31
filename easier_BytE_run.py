import collections
import csv
import json
import pickle
import argparse
import shutil

import pandas as pd

from dicee.executer import Execute
from dicee.config import Namespace
import os
import tiktoken

import custom_BytE_util
from custom_dice_executer import CustomExecute
from data_preperation import corpus_creation
from dicee.read_preprocess_save_load_kg import PreprocessKG


# noinspection PyTypeChecker
def find_exp_path(exp_path):
    exp_num = 0
    try:
        while True:
            exp = "Experiment%03d" % exp_num
            if "eval_report.json" not in os.listdir(os.path.join(exp_path, exp)):
                print(os.listdir(os.path.join(exp_path, exp)))
                shutil.rmtree(os.path.join(exp_path, exp))
                break
            exp_num += 1
    except FileNotFoundError:
        pass
    finally:
        exp_path = os.path.join(exp_path, "Experiment%03d" % exp_num)
    return exp_path, exp_num


if __name__ == '__main__':
    log_header = ["Dataset", "Model", "Linearization", "Embedding dimensions", "Learning rate", "num_epochs", "KG pool",
                  "Corpus input",
                  "Tie breaker",
                  "|VandR|", "relative Vocab size", "absolute vocab size", "Train H@10", "Train MRR", "Val H@10",
                  "Val MRR",
                  "Test H@10", "Test MRR", "exp_path"]
    experiment_log = collections.OrderedDict.fromkeys(log_header)

    parser = custom_BytE_util.CustomBytEParser()
    parser.add_argument("--forced_truncation", type=int, default=100,
                        help="Forced truncation truncates 100-x% of all training entities/relations.")
    parser.add_argument("--bpe_truncation", type=str, default="None",
                        choices=["None", "last_first", "asc_size", "desc_size", "asc_rank", "desc_rank"],
                        help="Order in which tokens are removed if entity/relation too long.")
    parser.add_argument("--bpe_with_RNN", type=str, default="Linear",
                        choices=["Linear", "RNN", "GRU", "LSTM"],
                        help="Linear function or RNN for token summaries.")
    parser.add_argument("--multiple_bpe_encodings", type=int, default=1,
                        help="Each edge is encoded n times, just with different tokens.")
    parser.add_argument("--multiple_bpe_loss", type=str, default="False",
                        help="True adds MSE to the loss, punishing different tokenizations of the same edge.")
    parser.add_argument("--model", help="KGE model used by Dicee", default="ComplEx", type=str)
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Number of dimensions for an embedding vector.')
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100, help='Highest number of epochs for training.')
    parser.add_argument("--min_epochs", type=int, help='Lowest number of epochs for training.')
    parser.add_argument("--vocab_size", help="Relative size of the token vocabulary.", type=float, default=1.0)
    parser.add_argument("--fix_missing", help="Whether Experiments above 0 are made.", type=int, default=-1)
    parser.add_argument("--exp_root", help="Directory for storing experiments.", type=str, default="Experiments")
    parser.add_argument("--base_seed", help="Random seed modifier, exp_num + seed is seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    paramspace = custom_BytE_util.ParameterSpace(vars(args))
    dataset_dir, vocab = paramspace.get_KG_and_voc_path(KG_path=args.KG_path)

    dicee_args = Namespace()
    dicee_args.dataset_dir = dataset_dir
    dicee_args.bpe_truncation = args.bpe_truncation
    dicee_args.forced_truncation = args.forced_truncation
    dicee_args.multiple_bpe_encodings = args.multiple_bpe_encodings
    dicee_args.multiple_bpe_loss = args.multiple_bpe_loss == "True"

    dicee_args.byte_pair_encoding = True
    dicee_args.trainer = "PL"
    dicee_args.scoring_technique = "KvsAll"
    dicee_args.eval_model = "train_val_test"
    dicee_args.bpe_with_RNN = args.bpe_with_RNN

    dicee_args.model = args.model
    dicee_args.embedding_dim = args.embedding_dim
    dicee_args.lr = args.lr
    dicee_args.num_epochs = args.num_epochs
    dicee_args.min_epochs = args.min_epochs

    element_count = len(
        corpus_creation(file_path="KGs", training_KG=args.training_KG, KG_pool="KG-custom", corpus_input="VandR"))
    if element_count < 256 * 4:
        args.vocab_size = 4 * args.vocab_size
    experiment_path = os.path.join(dataset_dir, paramspace.get_parameter_path()).replace(args.KG_path, args.exp_root)
    dicee_args.path_to_store_single_run, exp_num = find_exp_path(experiment_path)
    if -1 < args.fix_missing < exp_num:
        quit("We already have Experiments up to %d" % args.fix_missing)
    paramspace.exp_num = exp_num
    dicee_args.random_seed = paramspace.random_seed
    if args.KG_pool in custom_BytE_util.KG_pools:
        with open(vocab, "rb") as f:
            initial_ranks = pickle.load(f)
        ranks = dict(sorted(initial_ranks.items(), key=lambda x: x[1])[
                     :min([int(args.vocab_size * element_count), len(initial_ranks)])])
        enc = tiktoken.Encoding(name="Custom_tokenizer", mergeable_ranks=ranks, pat_str=r"[\s:_]|[^\s:_]+",
                                special_tokens={})
        print("Conducting Experiment %d on path %s" % (exp_num, dicee_args.path_to_store_single_run))
        print(dicee_args)
        executer = CustomExecute(dicee_args, vocabulary=enc)
    else:
        print("Conducting Experiment %d on path %s" % (exp_num, dicee_args.path_to_store_single_run))
        executer = Execute(dicee_args)
    executer.start()
    with open(os.path.join(dicee_args.path_to_store_single_run, "experiment_parameters.json"), "w") as f:
        json.dump(paramspace.get_parameter_dict(), f, indent=1, separators=(',', ': '))
