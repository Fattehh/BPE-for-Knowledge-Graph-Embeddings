import argparse
import json
import os
from itertools import chain

from custom_BytE_util import dir_path, ParameterSpace
import pandas as pd


def create_table(exp_path: str, extra_paths: list[str] = [], mode: str = "Train_Val_Test", sort_by: str = "Test",
                 sum_path="", ignore=["old"], name="") -> pd.DataFrame:
    flat_dfs = []
    modes = []
    for split in ["Train", "Val", "Test"]:
        if split in mode:
            modes.append(split)
    all_paths = [exp_path] + extra_paths
    for root, dirs, files in chain.from_iterable((os.walk(exp_path) for exp_path in all_paths)):
        if any([x in root for x in ignore + ["Summaries"]]):
            dirs[:] = []
        if "eval_report.json" in files:
            flat_dfs.append(ParameterSpace.from_path(root).create_df())
    df = pd.concat(flat_dfs, ignore_index=True)
    sorted_df = df[["parameters", *modes]].sort_values((sort_by, "_".join([sort_by, "MRR"])))
    if sum_path:
        os.makedirs(os.path.join(sum_path, "Summaries"), exist_ok=True)
        sorted_df.to_pickle(os.path.join(sum_path, "Summaries", "%ssorted_df.pkl" % name))
    else:
        os.makedirs(os.path.join(exp_path, "Summaries"), exist_ok=True)
        sorted_df.to_pickle(os.path.join(exp_path, "Summaries", "%ssorted_df.pkl" % name))
    return sorted_df


def create_time_table(exp_path: str, extra_paths: list[str] = [], sum_path="", ignore=["old"], name=""):
    param_list = []
    all_paths = [exp_path] + extra_paths
    for root, dirs, files in chain.from_iterable((os.walk(exp_path) for exp_path in all_paths)):
        if any([x in root for x in ignore + ["Summaries"]]):
            dirs[:] = []
        if "eval_report.json" in files:
            with open(os.path.join(root, 'report.json'), 'r') as f:
                v = json.load(f)
            param_list.append({**ParameterSpace.from_path(root).get_parameter_dict(), "Runtime": v["Runtime"]})
    rtime_df = pd.DataFrame(param_list)
    print(rtime_df.to_string())
    if sum_path:
        os.makedirs(os.path.join(sum_path, "Summaries"), exist_ok=True)
        rtime_df.to_pickle(os.path.join(sum_path, "Summaries", "%srtime_df.pkl" % name))
    else:
        os.makedirs(os.path.join(exp_path, "Summaries"), exist_ok=True)
        rtime_df.to_pickle(os.path.join(exp_path, "Summaries", "%srtime_df.pkl" % name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=dir_path)
    parser.add_argument("--extra_paths", nargs="*", default=[])
    parser.add_argument("--mode", help="String containing Train Val Test ", default="Train_Val_Test", type=str)
    parser.add_argument("--sort_by", choices=["Train", "Val", "Test"], default="Test", type=str)
    parser.add_argument("--sum_path", type=dir_path, default="")
    parser.add_argument("--ignore", nargs="*", default=[])
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()
    create_table(args.exp_path, args.extra_paths, args.mode, args.sort_by, args.sum_path, args.ignore, args.name)
    create_time_table(args.exp_path, args.extra_paths, args.sum_path, args.ignore, args.name)
