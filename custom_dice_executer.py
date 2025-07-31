import json
import logging
import time
import warnings
import tiktoken
from types import SimpleNamespace
import os
import datetime

from dicee import Execute
from pytorch_lightning import seed_everything

from typing import List
from dicee.read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
from dicee.knowledge_graph import KG
from dicee.evaluator import Evaluator
# Avoid
from dicee.static_preprocess_funcs import preprocesses_input_args
from dicee.trainer import DICE_Trainer
from dicee.static_funcs import timeit, continual_training_setup_executor, read_or_load_kg, load_json, store


class CustomExecute(Execute):
    # TODO Make this work for updated dicee, remove tokenizedKG and add truncate
    def __init__(self, args, continuous_training=False, vocabulary: tiktoken.Encoding = tiktoken.get_encoding("gpt2")):
        super().__init__(args, continuous_training)
        self.vocabulary = vocabulary

    def read_or_load_kg(self):
        print('*** Read or Load Knowledge Graph  ***')
        start_time = time.time()
        kg = KG(dataset_dir=self.args.dataset_dir,
                byte_pair_encoding=self.args.byte_pair_encoding,
                bpe_with_RNN=self.args.bpe_with_RNN,
                vocabulary=self.vocabulary,
                bpe_truncation=self.args.bpe_truncation, forced_truncation=self.args.forced_truncation,
                multiple_bpe_encodings=self.args.multiple_bpe_encodings, multiple_bpe_loss=self.args.multiple_bpe_loss,
                padding=True if self.args.byte_pair_encoding and self.args.model != "BytE" else False,
                add_noise_rate=self.args.add_noise_rate,
                sparql_endpoint=self.args.sparql_endpoint,
                path_single_kg=self.args.path_single_kg,
                add_reciprical=self.args.apply_reciprical_or_noise,
                eval_model=self.args.eval_model,
                read_only_few=self.args.read_only_few,
                sample_triples_ratio=self.args.sample_triples_ratio,
                path_for_serialization=self.args.full_storage_path,
                path_for_deserialization=self.args.path_experiment_folder if hasattr(self.args,
                                                                                     'path_experiment_folder') else None,
                backend=self.args.backend,
                training_technique=self.args.scoring_technique)
        print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
        # (2) Share some info about data for easy access.
        print(kg.description_of_input)
        return kg


logging.getLogger('pytorch_lightning').setLevel(0)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
