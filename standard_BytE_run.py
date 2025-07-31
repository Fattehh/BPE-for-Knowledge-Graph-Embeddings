import collections
import csv
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


def find_exp_path(exp_path):
    exp_num = 0
    try:
        while True:
            exp = "Experiment%03d" % exp_num
            if "eval_report.json" not in os.listdir(os.path.join(experiment_path, exp)):
                print(os.listdir(os.path.join(experiment_path, exp)))
                shutil.rmtree(os.path.join(experiment_path, exp))
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
    parser.add_argument("--custom_tokenization", help="BytE with custom tokenization", action="store_true")
    parser.add_argument("--bpe_truncation",
                        choices=["", "None", "last_first", "asc_size", "desc_size", "asc_rank", "desc_rank"],
                        default="None",
                        help="Currently only avail. for KGE implemented within dice-embeddings.")
    parser.add_argument("--bpe_with_RNN", choices=["", "Linear", "RNN", "GRU", "LSTM"], default="Linear",
                        help="Linear function or RNN for token summaries.")
    parser.add_argument("--forced_truncation", type=int, default=100,
                        help="Forced truncation truncates 1-x of all training entities/relations.")
    parser.add_argument("--multiple_bpe_encodings", type=int, default=1,
                        help="Each edge is encoded n times, just with different tokens.")
    parser.add_argument("--model", help="KGE model used by Dicee", default="ComplEx", type=str)
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Number of dimensions for an embedding vector.')
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100, help='Number of epochs for training. ')
    parser.add_argument("--vocab_size", help="Relative size of the token vocabulary.", type=float, default=1.0)
    parser.add_argument("--fix_missing", help="Whether Experiments above 0 are made.", type=int, default=-1)
    parser.add_argument("--benchmark", help="Experiments are done in a seperate folder.")
    parser.add_argument("--seed", help="Random seed modifier, exp_num + seed is seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    if args.benchmark:
        exp_root = args.benchmark
    else:
        exp_root = "Experiments"
    dicee_args = Namespace()
    if args.bpe_truncation == "":
        dicee_args.bpe_truncation = "None"
    else:
        dicee_args.bpe_truncation = args.bpe_truncation
    dicee_args.forced_truncation = args.forced_truncation
    dicee_args.multiple_bpe_encodings = args.multiple_bpe_encodings
    for root, dirs, files in os.walk(args.KG_path):
        if args.training_KG in dirs:
            dicee_args.dataset_dir = os.path.join(root, args.training_KG)
    dicee_args.byte_pair_encoding = True
    dicee_args.trainer = "PL"
    dicee_args.scoring_technique = "KvsAll"
    dicee_args.eval_model = "train_val_test"
    dicee_args.bpe_with_RNN = args.bpe_with_RNN
    print(dicee_args)
    experiment_log["Model"] = dicee_args.model = args.model
    experiment_log["Linearization"] = dicee_args.bpe_with_RNN
    experiment_log["Embedding dimensions"] = dicee_args.embedding_dim = args.embedding_dim
    experiment_log["Learning rate"] = dicee_args.lr = args.lr
    experiment_log["num_epochs"] = dicee_args.num_epochs = args.num_epochs

    experiment_log["Dataset"] = args.training_KG
    experiment_log["|VandR|"] = element_count = len(corpus_creation(file_path="KGs", training_KG=args.training_KG,
                                                                    KG_pool="KG-custom", corpus_input="VandR"))
    if element_count < 256 * 4:
        args.vocab_size = 4 * args.vocab_size
    vocabulary_path = dicee_args.dataset_dir.replace("KGs", "Vocabularies")
    vocabularies = []
    if args.multiple_bpe_encodings:
        model_dir = "_".join([dicee_args.model, dicee_args.bpe_with_RNN, "multi"])
    elif args.bpe_truncation:
        model_dir = "_".join([dicee_args.model, dicee_args.bpe_with_RNN, str(args.forced_truncation), dicee_args.bpe_truncation])
    elif args.bpe_with_RNN:
        model_dir = "_".join([dicee_args.model, dicee_args.bpe_with_RNN])
    else:
        model_dir = args.model
    if args.custom_tokenization:
        experiment_log["KG pool"] = args.KG_pool
        experiment_log["Corpus input"] = args.corpus_input
        experiment_log["Tie breaker"] = args.tie_breaker
        experiment_log["relative Vocab size"] = args.vocab_size
        if "KG-pretrained" == args.KG_pool:
            vocab = os.path.join(os.path.dirname(vocabulary_path), "KG-pretrained", args.corpus_input,
                                 args.tie_breaker + "_vocab.pkl")
        else:
            vocab = os.path.join(str(vocabulary_path), args.KG_pool, args.corpus_input, args.tie_breaker + "_vocab.pkl")
        try:
            with open(vocab, "rb") as f:
                initial_ranks = pickle.load(f)
                ranks = dict(sorted(initial_ranks.items(), key=lambda x: x[1])[
                             :min([int(args.vocab_size * element_count), len(initial_ranks)])])
        except FileNotFoundError:
            raise FileNotFoundError("No Vocabulary at %s" % vocab)
        experiment_log["absolute Vocab size"] = len(ranks)
        print(vocab)
        experiment_path = vocab.replace("Vocabularies", exp_root).replace(".pkl", "")
        if "KG-pretrained" == args.KG_pool:
            idx = experiment_path.index("KG-pretrained")
            experiment_path = os.path.join(experiment_path[:idx], args.training_KG, experiment_path[idx:])
        experiment_path = os.path.join(str(experiment_path), str(args.vocab_size), model_dir,
                                       str(dicee_args.embedding_dim), str(dicee_args.lr))
        with open(vocab, "rb") as f:
            initial_ranks = pickle.load(f)
            ranks = dict(sorted(initial_ranks.items(), key=lambda x: x[1])[
                         :min([int(args.vocab_size * element_count), len(initial_ranks)])])
        enc = tiktoken.Encoding(name="Custom_tokenizer", mergeable_ranks=ranks, pat_str=r"[\s:_]|[^\s:_]+",
                                special_tokens={})
        dicee_args.path_to_store_single_run, experiment_num = find_exp_path(experiment_path)
        dicee_args.random_seed = experiment_num + args.seed
        if -1 < args.fix_missing < experiment_num:
            quit("We already have Experiments up to %d" % args.fix_missing)
        print("Conducting Experiment %d on path %s" % (experiment_num, dicee_args.path_to_store_single_run))
        # TODO try to get rid of configuration.json containing vocabulary
        print(dicee_args)
        executer = CustomExecute(dicee_args, vocabulary=enc)
    else:
        experiment_log["KG pool"] = "original_BytE"
        experiment_path = dicee_args.dataset_dir.replace("KGs", exp_root)
        experiment_path = os.path.join(str(experiment_path), "original_BytE", model_dir,
                                       str(dicee_args.embedding_dim), str(dicee_args.lr))
        dicee_args.path_to_store_single_run, experiment_num = find_exp_path(experiment_path)
        dicee_args.random_seed = experiment_num + args.seed
        if -1 < args.fix_missing < experiment_num:
            quit("We already have Experiments up to %d" % args.fix_missing)
        print("Conducting Experiment %d on path %s" % (experiment_num, dicee_args.path_to_store_single_run))
        executer = Execute(dicee_args)
    executer.start()
    experiment_log["exp_path"] = dicee_args.path_to_store_single_run
    print(os.listdir(dicee_args.path_to_store_single_run))
    df = pd.read_json(os.path.join(dicee_args.path_to_store_single_run, "eval_report.json"))
    experiment_log["Train H@10"] = df["Train"].loc["H@10"]
    experiment_log["Train MRR"] = df["Train"].loc["MRR"]
    experiment_log["Val H@10"] = df["Val"].loc["H@10"]
    experiment_log["Val MRR"] = df["Val"].loc["MRR"]
    experiment_log["Test H@10"] = df["Test"].loc["H@10"]
    experiment_log["Test MRR"] = df["Test"].loc["MRR"]
    os.makedirs("Experiments/Summaries", exist_ok=True)
    try:
        with open("Experiments/Summaries/KGELog.csv", 'x', newline="") as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            writer.writerow(log_header)
            writer.writerow(list(experiment_log.values()))
    except FileExistsError:
        with open("Experiments/Summaries/KGELog.csv", 'a', newline="") as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            writer.writerow(list(experiment_log.values()))
