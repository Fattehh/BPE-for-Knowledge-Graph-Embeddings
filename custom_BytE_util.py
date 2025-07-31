import argparse
import math
import pickle
from collections import OrderedDict
import json
import os
import pathlib

import pandas as pd

KG_pools = ["KG-custom", "KG-pretrained", "KG-finetuned"]
corpus_inputs = ["VandR", "E", "randomWalks"]
tie_breakers = ["default", "ascending_size", "descending_size"]


def dir_path(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def shell_loop_from_loops(commands: list[str], loop_names: list[str], loops: list[list[str]],
                          indent: int = 0) -> str:
    if len(loops) == 0:
        if loop_names:
            cmds = "\n".join(
                [" " * indent + "%s " % cmd + " ".join(
                    ["--%s $%s" % (name, name) for name in loop_names if name != "training_kg"]) for cmd in commands])
        else:
            cmds = "\n".join([" " * indent + "%s" % cmd for cmd in commands])
        return cmds + "\n"
    else:
        shell_loop = " " * indent + " ".join(["for", loop_names[-len(loops)], "in", *loops[0]]) + "; do\n"
        shell_loop += shell_loop_from_loops(commands, loop_names, loops[1:], indent + 2)
        shell_loop += " " * indent + "done\n"
    return shell_loop


def is_known_KG(KG_path: str, training_KG: str):
    for root, dirs, files in os.walk(KG_path):
        if training_KG in dirs:
            return os.path.join(root, training_KG)
    raise FileNotFoundError("KG %s not in %s" % (training_KG, KG_path))


class ParameterSpace:
    keys = ["training_KG", "KG_pool", "corpus_input", "tie_breaker", "vocab_size", "model", "embedding_dim",
            "lr", "num_epochs", "min_epochs", "base_seed", "forced_truncation", "bpe_truncation", "bpe_with_RNN",
            "multiple_bpe_encodings"]

    @classmethod
    def from_path(cls, path: str):
        if not path.endswith("experiment_parameters.json"):
            path = os.path.join(path, "experiment_parameters.json")
        try:
            with open(path) as jsonfile:
                pspace = cls(json.load(jsonfile))
                pspace.exp_path = os.path.dirname(path)
                return pspace
        except:
            raise FileNotFoundError("No experiment at %s" % path)

    def __init__(self, kwargs: dict[str, str]):
        self.training_KG = kwargs.get("training_KG")
        self.KG_pool = kwargs.get("KG_pool", "")
        self.corpus_input = kwargs.get("corpus_input", "")
        self.tie_breaker = kwargs.get("tie_breaker", "")
        if self.tie_breaker.endswith("_vocab"):
            self.tie_breaker = self.tie_breaker[:-6]
        self.vocab_size = str(kwargs.get("vocab_size", ""))
        self.model = kwargs.get("model", "")
        self.embedding_dim = kwargs.get("embedding_dim", "")
        self.lr = str(kwargs.get("lr", ""))
        self.num_epochs = str(kwargs.get("num_epochs", ""))
        self.min_epochs = str(kwargs.get("min_epochs", ""))

        self.bpe_with_RNN = kwargs.get("bpe_with_RNN", "")
        self.forced_truncation = kwargs.get("forced_truncation", "")
        self.bpe_truncation = kwargs.get("bpe_truncation", "")
        self.multiple_bpe_encodings = kwargs.get("multiple_bpe_encodings", "")
        self.multiple_bpe_loss = kwargs.get("multiple_bpe_loss", "False")

        self._exp_root = kwargs.get("exp_root", "")
        self._exp_num = str(kwargs.get("exp_num", 0))
        self.fix_missing = str(kwargs.get("fix_missing", -1))
        self.base_seed = str(kwargs.get("base_seed", 0))
        self.exp_path = False
        self.parameters = kwargs
        self.parameters.pop("random_seed", None)
        self.parameters.pop("exp_num", None)

    def create_df(self):
        if not self.exp_path:
            for root, dirs, files in os.walk(self._exp_root):
                if os.path.isdir(os.path.join(root, self.training_KG, self.get_parameter_path(),
                                              "Experiment%03d" % int(self._exp_num))):
                    self.exp_path = os.path.join(root, self.training_KG, self.get_parameter_path(),
                                                 "Experiment%03d" % int(self._exp_num))
                    break
            else:
                raise FileNotFoundError("Experiment not found at %s" % self.get_parameter_path())
        eval_df = pd.read_json(os.path.join(self.exp_path, "eval_report.json"))
        param_df = pd.DataFrame(self.parameters, index=[0])
        param_df.columns = pd.MultiIndex.from_product([["parameters"], param_df.columns])
        result = (flat_df := eval_df.unstack().to_frame().T).set_axis([f"{i}_{j}" for i, j in flat_df.columns], axis=1)
        result.columns = [[x[0] for x in result.columns.str.split("_")], result.columns]
        return param_df.join(result)

    def get_parameter_dict(self) -> dict[str, str]:
        return self.parameters

    def get_KG_and_voc_path(self, KG_path):
        KG_dir = is_known_KG(KG_path, self.training_KG)
        try:
            if self.KG_pool == "original_BytE":
                return KG_dir, None
            vocabulary_path = KG_dir.replace("KGs", "Vocabularies")
        except:
            raise FileNotFoundError("Dataset %s not found in %s" % (self.training_KG, KG_path))
        if "KG-pretrained" == self.KG_pool:
            vocab = os.path.join(os.path.dirname(vocabulary_path), "KG-pretrained", self.corpus_input,
                                 self.tie_breaker + "_vocab.pkl")
        else:
            vocab = os.path.join(str(vocabulary_path), self.KG_pool, self.corpus_input, self.tie_breaker + "_vocab.pkl")
        try:
            with open(vocab, "rb") as f:
                pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("No Vocabulary at %s" % vocab)
        return KG_dir, vocab

    def get_parameter_path(self):
        last_dir = self.bpe_with_RNN
        if self.forced_truncation:
            last_dir = "_".join([last_dir, "forced%d" % self.forced_truncation])
        else:
            last_dir = "_".join([last_dir, "trunc"])
        if self.bpe_truncation in ["", "None"]:
            last_dir = "_".join([last_dir, "None"])
        else:
            last_dir = "_".join([last_dir, self.bpe_truncation])
        if int(self.multiple_bpe_encodings) > 1:
            last_dir = "_".join([last_dir, "multi%s%s" % (self.multiple_bpe_encodings, self.multiple_bpe_loss)])
        if self.KG_pool == "original_BytE":
            return os.path.join(self.KG_pool,
                                "_".join([self.model, "emb%s" % self.embedding_dim, "lr%s" % self.lr,
                                          "epoch%s" % self.num_epochs]), last_dir)
        else:
            return os.path.join("_".join([self.KG_pool, self.corpus_input, self.tie_breaker, self.vocab_size]),
                                "_".join([self.model, "emb%s" % self.embedding_dim, "lr%s" % self.lr,
                                          "epoch%s" % self.num_epochs]), last_dir)

    def python_command(self, skip: list[str] = []):
        py_str = " ".join(["--%s %s" % (k, str(v)) for k, v in self.parameters.items() if str(v) and k not in skip])
        py_str = " ".join([".venv/bin/python easier_BytE_run.py", py_str]).replace("--training_KG ", "").replace(
            "--KG_path ", "")
        return py_str

    @property
    def exp_num(self):
        return self._exp_num

    @exp_num.setter
    def exp_num(self, exp_num):
        self.parameters["exp_num"] = self._exp_num = str(exp_num)
        self.parameters["random_seed"] = self.random_seed = str(int(self.base_seed) + int(exp_num))

    @property
    def exp_root(self):
        return self._exp_root

    @exp_root.setter
    def exp_root(self, exp_root):
        self.parameters["exp_root"] = self._exp_root = str(exp_root)

    @property
    def base_seed(self):
        return self._base_seed

    @base_seed.setter
    def base_seed(self, base_seed):
        self._base_seed = base_seed
        self.random_seed = str(int(self.base_seed) + int(self._exp_num))

    @classmethod
    # @deprecated("Only use on old experiments")
    def from_old_path(cls, path: str):
        with open(os.path.join(path, "configuration.json")) as jsonfile:
            num_epochs = json.load(jsonfile)["num_epochs"]
        parameter_path_parts = list(pathlib.Path(path).parts)
        for i in range(len(parameter_path_parts)):
            if parameter_path_parts[i] in KG_pools:
                params = parameter_path_parts[i - 1:i + 7]
                break
            elif parameter_path_parts[i] == "original_BytE":
                params = parameter_path_parts[i - 1:i + 4]
                params[2:2] = [""] * 3
                break
        keys = ["training_KG", "KG_pool", "corpus_input", "tie_breaker", "vocab_size", "model", "embedding_dim",
                "lr", "num_epochs"]
        kwargs = OrderedDict(zip(keys, [*params, num_epochs]))
        if kwargs.get("tie_breaker", "").endswith("_vocab"):
            kwargs["tie_breaker"] = kwargs["tie_breaker"][:-6]
        result = cls(kwargs)
        # Todo better error catching
        result.exp_root = os.path.join(*parameter_path_parts[:i - 1])
        if "Experiment" in parameter_path_parts[-1]:
            result.exp_num = int(parameter_path_parts[-1][-3:])
        return result

    # @deprecated("Only use on old experiments")
    def create_old_df(self):
        exp_path = os.path.join(self.exp_root, self.get_old_parameter_path(), "Experiment%03d" % self._exp_num)
        eval_df = pd.read_json(os.path.join(exp_path, "eval_report.json"))
        param_df = pd.DataFrame(self.parameters, index=[0])
        param_df.columns = pd.MultiIndex.from_product([["parameters"], param_df.columns])
        result = (flat_df := eval_df.unstack().to_frame().T).set_axis([f"{i}_{j}" for i, j in flat_df.columns], axis=1)
        result.columns = [[x[0] for x in result.columns.str.split("_")], result.columns]
        return result

    # @deprecated("Only use on old experiments")
    def get_old_parameter_path(self):
        if self.KG_pool == "original_BytE":
            return str(os.path.join(*list(self.parameters.values())[:-1]))
        else:
            return str(os.path.join(*list(self.parameters.values())[:-1])).replace(self.tie_breaker,
                                                                                   self.tie_breaker + "_vocab")


class CustomBytEParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("KG_path", help="path to KGs", default="KGs/", type=dir_path)
        self.add_argument("training_KG", help="Dataset used to train KGE.", type=str)
        self.add_argument("--KG_pool", help="Dataset allowed to build token vocabulary.", type=str,
                          choices=[*KG_pools, "original_BytE"], default="original_BytE")
        self.add_argument("--corpus_input", help="Part of the KG used to build token vocabulary", type=str,
                          choices=[*corpus_inputs, "None"], default="None")
        self.add_argument("--tie_breaker", help="Tie-breaker used for equal length tokens", type=str,
                          choices=[*tie_breakers, "None"], default="None")


class Benchmarker:
    explore_dict = {"forced_truncation": ["90", "95", "100"],
                    "multiple_bpe_encodings": ["1", "2", "5", "10"],
                    "bpe_with_RNN": ["Linear", "RNN", "GRU", "LSTM"],
                    "KG_pool": KG_pools,
                    "corpus_input": corpus_inputs,
                    "tie_breaker": ["ascending_size", "descending_size"],
                    "vocab_size": ["0.5", "1", "2"], }

    def __init__(self, name: str, paramspaces: list[ParameterSpace], modes: dict[str, list[str]],
                 sep_modes: list[dict[str, list[str]]], repetitions: int, training_KGs: list[str] = []):
        self.name = name
        self.paramspaces = paramspaces
        for KG in training_KGs:
            is_known_KG("KGs", KG)
        for ps in paramspaces:
            ps.exp_root = os.path.join("automated_scripts", self.name)
            for key in list(modes.keys()) + [key for sep_mode in sep_modes for key in sep_mode.keys()]:
                assert key in ps.get_parameter_dict().keys(), key
        self.modes = modes
        self.sep_modes = sep_modes
        self.training_KGs = training_KGs
        self.repetitions = repetitions
        for mode in self.modes.items():
            if "explore" in mode[1]:
                self.modes[mode[0]] = self.explore_dict[mode[0]]
        for sep_mode in self.sep_modes:
            for mode in sep_mode.items():
                if "explore" in mode[1]:
                    sep_mode[mode[0]] = self.explore_dict[mode[0]]
        self.shared_experiments = math.prod([len(mode) for mode in self.modes.values()])
        self.unshared_experiments = max(len(self.training_KGs), 1) * max(sum(
            [math.prod([len(mode) for mode in sep_mode.values()]) for sep_mode in self.sep_modes]), 1) * len(
            paramspaces)
        self.total_experiments = self.shared_experiments * self.unshared_experiments

    def create_benchmark_scripts(self):
        # Set lower the limit according to the number of unshared experiments and training_KGs
        print(self.name)
        loop_limit = (int(100 / self.unshared_experiments), int(300 / self.unshared_experiments))
        outer_modes, inner_modes = self.loopsplitter(self.modes, loop_limit)
        shell_command = ["sbatch " + os.path.join("automated_scripts", self.name, "benchmark.sh")]
        new_params = list(self.modes.keys()) + [params for sep_mode in self.sep_modes for params in sep_mode.keys()]
        if self.training_KGs:
            for ps in self.paramspaces:
                ps.parameters["training_KG"] = ps.training_KG = '"$training_kg"'
            job_commands = [ps.python_command(new_params) for ps in self.paramspaces]
        else:
            job_commands = [ps.python_command(new_params) for ps in self.paramspaces]
        shell_command = [" ".join([shell_command[0], job_command]) for job_command in job_commands]
        shells = []
        # Create loops for training KGs
        if self.training_KGs:
            # Fills training_KGs loop with outer loops
            if not self.sep_modes:
                shells = [shell_loop_from_loops(shell_command, list(outer_modes.keys()), list(outer_modes.values()), 2)]
            # Fills training_KGs loop with unshared loops filled with outer loops
            else:
                for sep_mode in self.sep_modes:
                    shell = shell_loop_from_loops(shell_command, list(sep_mode.keys()) + list(outer_modes.keys()),
                                                  list(sep_mode.values()) + list(outer_modes.values()), 2)
                    shells.append(shell)
            shells = ["for training_kg in %s; do\n" % (" ".join(self.training_KGs))] + shells + ["done"]
        else:
            # Outer loops
            if not self.sep_modes:
                shells = [shell_loop_from_loops(shell_command, list(outer_modes.keys()), list(outer_modes.values()), 2)]
            # unshared loops filled with outer loops
            else:
                for sep_mode in self.sep_modes:
                    shell = shell_loop_from_loops(shell_command, list(sep_mode.keys()) + list(outer_modes.keys()),
                                                  list(sep_mode.values()) + list(outer_modes.values()))
                    shells.append(shell)
        os.makedirs(os.path.join("automated_scripts", self.name), exist_ok=True)
        # Save outer bash script
        with open(os.path.join("automated_scripts", self.name, "benchmarks.sh"), "w", newline="") as f:
            f.write("#!/bin/bash\n")
            f.write("# This script hands in %d batch jobs containing %d experiments total\n" % (
                math.prod([len(mode) for mode in outer_modes.values()]) * self.unshared_experiments,
                self.total_experiments * self.repetitions))
            f.writelines(shells)

        # Innter bash script
        inner_experiments = math.prod([len(mode) for mode in inner_modes.values()]) * self.repetitions
        time = max([1, min([int(inner_experiments / 2), 8])])
        job_commands = ['"$@"']
        job = shell_loop_from_loops(job_commands, list(inner_modes.keys()), list(inner_modes.values()), 2)
        with open(os.path.join("automated_scripts", self.name, "benchmark.sh"), "w", newline="") as f:
            f.write(
                '#!/bin/bash\n#SBATCH --time=0%d:00:00\n#SBATCH --partition=accelerated\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-gpu=1\n#SBATCH -o ./.Report/KGE/%s.%%j.out\n#STDOUT\necho "$0" "$@"' % (
                    time, self.name))
            f.write(
                '\n# This script conducts %d experiments.\nstart=$(date +%%s)\nlength=$((%d*60))\nmintime=$((%d*60))\nfor run in %s; do\n  dif=$(($length - $(date "+%%s") + $start ))\n  if [ $dif -lt $mintime ]; then\n    echo "Out of time"\n    exit 1\n  fi\n' % (
                    inner_experiments, time*60, min(60*time//self.repetitions, 30*time)," ".join([str(i) for i in range(self.repetitions)])))
            f.write(job)
            f.write("done")
        debug_commands = [ps.python_command(list(self.modes.keys())) for ps in self.paramspaces]
        debug = shell_loop_from_loops(debug_commands, list(self.modes.keys()), list(self.modes.values()))
        debug = debug.replace("bin/python", "Scripts/python.exe").replace("\\", "/")
        with open(os.path.join("automated_scripts", self.name, "debug.sh"), "w", newline="") as f:
            f.write("#!/bin/bash\n")
            f.write("# This script is to check if experiments run bugfree\n")
            f.write(debug)

    @staticmethod
    def loopsplitter(loop_dict: dict[str, list[str]], outer_limit: tuple[int, int] = (100, 300)) -> (
            dict[str, list[str]], dict[str, list[str]]):
        runs_per_job = 1
        outer_loop = []
        for loop in loop_dict.items():
            if runs_per_job > outer_limit[0] or runs_per_job * len(loop[1]) > outer_limit[1]:
                break
            else:
                runs_per_job *= len(loop[1])
                outer_loop.append(loop)
        inner_loop = {loop[0]: loop[1] for loop in loop_dict.items() if loop not in outer_loop}
        outer_loop = {loop[0]: loop[1] for loop in outer_loop}
        return outer_loop, inner_loop

    @staticmethod
    def create_tokenization_scripts(name: str, training_kgs: list[str], pretrain_kgs: list[list[str]]):
        commands = [" ".join(
            ["sbatch", os.path.join("automated_scripts", name, "tokenization.sh"), "KGs", training_kg, "--pretrain_KGs",
             " ".join(pretrain_kg), "--logging"]) for training_kg, pretrain_kg in zip(training_kgs, pretrain_kgs)]
        jobs = ['.venv/bin/python tokenization_script.py "$@"']
        loop_dict1 = {"KG_pool": ["KG-custom"], "corpus_input": corpus_inputs,
                      "tie_breaker": ["ascending_size", "descending_size"]}
        loop_dict2 = {"KG_pool": ["KG-pretrained", "KG-finetuned"], "corpus_input": ["VandR"],
                      "tie_breaker": ["ascending_size", "descending_size"]}
        loop1 = shell_loop_from_loops(commands, list(loop_dict1.keys()), list(loop_dict1.values()))
        loop2 = shell_loop_from_loops(commands, list(loop_dict2.keys()), list(loop_dict2.values()))
        os.makedirs(os.path.join("automated_scripts", name), exist_ok=True)
        with open(os.path.join("automated_scripts", name, "tokenizations.sh"), "w", newline="") as f:
            f.write("\n# This script does %d tokenization jobs.\n" % (len(commands) * (3 * 2 + 2 * 2)))
            f.write(loop1)
            f.write(loop2)
        with open(os.path.join("automated_scripts", name, "tokenization.sh"), "w", newline="") as f:
            f.write(
                '#!/bin/bash\n#SBATCH --cpus-per-task=1\n#SBATCH --time=12:00:00\n#SBATCH --ntasks=1\n#SBATCH --partition=cpuonly\n#SBATCH -o ./.Report/tokenization/Tokenize.%j.out # STDOUT\n')
            f.write("\n".join(jobs))


class EvalReport:
    csv_header = ["Dataset", "Model", "Embedding dimensions", "Learning rate", "KG pool", "Corpus input", "Tie breaker",
                  "Vocab size", "Train H@10", "Train MRR", "Test H@10", "Test MRR"]

    def __init__(self, dataset: str, model: str, emb_dims: int, lr: float, KG_pool: str, corpus_input: str,
                 tie_breaker: str, vocab_size: float, train_h10: float, train_mrr: float, test_h10: float,
                 test_mrr: float):
        self.dataset = dataset
        self.model = model
        self.emb_dims = emb_dims
        self.lr = float(lr)
        self.KG_pool = KG_pool
        self.corpus_input = corpus_input
        self.tie_breaker = tie_breaker
        if vocab_size:
            self.vocab_size = float(vocab_size)
        else:
            self.vocab_size = vocab_size
        self.train_h10 = float(train_h10)
        self.train_mrr = float(train_mrr)
        self.test_h10 = float(test_h10)
        self.test_mrr = float(test_mrr)

    def create_csv(self):
        return [self.dataset, self.model, str(self.emb_dims), str(self.lr), self.KG_pool, self.corpus_input,
                self.tie_breaker, str(self.vocab_size), str(self.train_h10), str(self.train_mrr), str(self.test_h10),
                str(self.test_mrr)]
