import argparse
import os.path
import pickle

import pandas as pd

from custom_BytE_util import ParameterSpace, dir_path


def best_parameters_from_path(exp_path: dir_path, kept_params: list[str] = [], gpt: bool = True) -> list[
    ParameterSpace]:
    if gpt:
        prefix = "gpt2_" + "_".join(kept_params) + "_"
    elif kept_params:
        prefix = "_".join(kept_params) + "_"
    else:
        prefix = ""
    kept_params = [("parameters", param) for param in kept_params]
    df = pd.read_pickle(os.path.join(exp_path, "Summaries", "sorted_df.pkl"))
    df.to_csv(os.path.join(exp_path, "Summaries", os.path.basename(exp_path) + "_sorted_df.csv"))
    if gpt:
        sorted_custom_df = df.loc[df[("parameters", "KG_pool")] != "original_BytE"][["parameters", "Val"]].sort_values(
            ("Val", "Val_MRR"))
        sorted_orig_df = df.loc[df[("parameters", "KG_pool")] == "original_BytE"][["parameters", "Val"]].sort_values(
            ("Val", "Val_MRR"))
        maxed_custom_df = sorted_custom_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
        maxed_orig_df = sorted_orig_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
        maxed_df = pd.concat([maxed_custom_df, maxed_orig_df])
    else:
        sorted_df = df[["parameters", "Val"]].sort_values(("Val", "Val_MRR"))
        maxed_df = sorted_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
    maxed_df = maxed_df.sort_values([("parameters", "training_KG"), ("Val", "Val_MRR")], ascending=False)
    # print(maxed_df.to_string())
    # print(maxed_df.drop_duplicates([("parameters", "training_KG")], keep="first"))

    rows = maxed_df["parameters"].values.tolist()
    columns = []
    for col in maxed_df.columns:
        if "parameters" in col:
            columns.append(col[1])
    rows = [{k: v for k, v in zip(columns, row)} for row in rows]
    parameters_list = []
    commands = []
    for row in rows:
        parameters = ParameterSpace(row)
        parameters_list.append(parameters)
        commands.append(parameters.python_command())
    maxed_df.to_csv(os.path.join(exp_path, "Summaries", "maxed_df.csv"))
    with open(os.path.join(exp_path, "Summaries", prefix + "Best_parameters.pkl"), "wb") as f:
        pickle.dump(parameters_list, f)
    return parameters_list


def select_best_parameters(res_path: dir_path, prefix: str, sum_df: pd.DataFrame, kept_params: list[str] = [],
                           gpt: bool = True) -> list[ParameterSpace]:
    kept_params = [("parameters", param) for param in kept_params]
    sum_df.to_csv(os.path.join(res_path, "Summaries", prefix + "_sorted_df.csv"))
    if gpt:
        sorted_custom_df = sum_df.loc[sum_df[("parameters", "KG_pool")] != "original_BytE"][
            ["parameters", "Val"]].sort_values(
            ("Val", "Val_MRR"))
        sorted_orig_df = sum_df.loc[sum_df[("parameters", "KG_pool")] == "original_BytE"][
            ["parameters", "Val"]].sort_values(
            ("Val", "Val_MRR"))
        maxed_custom_df = sorted_custom_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
        maxed_orig_df = sorted_orig_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
        maxed_df = pd.concat([maxed_custom_df, maxed_orig_df])
    else:
        sorted_df = sum_df[["parameters", "Val"]].sort_values(("Val", "Val_MRR"))
        maxed_df = sorted_df.drop_duplicates([("parameters", "training_KG"), *kept_params], keep="last")
    maxed_df = maxed_df.sort_values([("parameters", "training_KG"), ("Val", "Val_MRR")], ascending=False)
    # print(maxed_df.to_string())
    # print(maxed_df.drop_duplicates([("parameters", "training_KG")], keep="first"))

    rows = maxed_df["parameters"].values.tolist()
    columns = []
    for col in maxed_df.columns:
        if "parameters" in col:
            columns.append(col[1])
    rows = [{k: v for k, v in zip(columns, row)} for row in rows]
    parameters_list = []
    commands = []
    for row in rows:
        parameters = ParameterSpace(row)
        parameters_list.append(parameters)
        commands.append(parameters.python_command())
    maxed_df.to_csv(os.path.join(res_path, "Summaries", prefix + "_maxed_df.csv"))
    with open(os.path.join(res_path, "Summaries", prefix + "_Best_parameters.pkl"), "wb") as f:
        pickle.dump(parameters_list, f)
    return parameters_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=dir_path)
    parser.add_argument("--kept_params", type=str, nargs="+")
    args = parser.parse_args()
    best_parameters_from_path(args.exp_path)
