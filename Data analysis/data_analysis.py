import argparse
import csv
import os.path

import pandas as pd


def parameter_analysis(dataframe: pd.DataFrame, parameters: list[str]) -> (pd.DataFrame, pd.DataFrame):
    # First rank each experiment according to (Val, MRR)
    # TODO Best rank with pct rank method.
    param_columns = [("parameters", parameter) for parameter in parameters]
    df = dataframe.loc[:, [("parameters", "Dataset"), *param_columns, ("Val", "Val_MRR")]]
    df[("Val", "min ranks")] = df.groupby([("parameters", "Dataset")])[[("Val", "Val_MRR")]].rank(method="min",
                                                                                              ascending=False)
    # print(pd.DataFrame(df.value_counts([*param_columns])).reset_index().to_string(index=False))
    df_min_rank_groups = df.groupby([("parameters", "Dataset"), *param_columns])[[("Val", "min ranks")]]
    # Gives the best rank for each parameter value and dataset -> Which parameter value got the first place?
    param_ranks = df_min_rank_groups.min()
    # Averages ranks for each dataset -> Which parameter value is generally good/bad for a given dataset?
    param_ranks[("Val", "Mean rank")] = df_min_rank_groups.mean()
    param_ranks.sort_values([("parameters", "Dataset"), ("Val", "min ranks")], inplace=True)
    param_ranks.rename(columns={"min ranks": "Best rank"}, inplace=True)
    # Now averages of the above, First, which parameter value has the most first places?
    dataset_avg_ranks = param_ranks.groupby([*param_columns])[[("Val", "Best rank")]].mean()
    # Second, Which parameter value is generally good/bad?
    dataset_avg_ranks[("Val", "Mean² rank")] = param_ranks.groupby([*param_columns])[
        [("Val", "Mean rank")]].mean()
    dataset_avg_ranks.rename(columns={"Best rank": "Mean best rank"}, inplace=True)
    dataset_avg_ranks.sort_values([("Val", "Mean best rank")], inplace=True)
    return param_ranks, dataset_avg_ranks


def create_means_from_summaries(summary_dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :param test_dataframe: df created from a summaries.csv
    :return: means sorted by Dataset and Test MRR
    """

    parameters = summary_dataframe["parameters"].columns
    means_df = summary_dataframe.groupby([("parameters", parameter) for parameter in parameters]).mean()
    means_df = means_df.reset_index().sort_values(
        [("parameters", "Dataset"), ("Test", "Test_MRR")], ascending=False)
    means_df[("parameters", "num_epochs")] = means_df[("parameters", "num_epochs")].astype(int)
    means_df[("parameters", "Embedding dimensions")] = means_df[("parameters", "Embedding dimensions")].astype(int)
    return means_df


def clean_benchmarks(means_df: pd.DataFrame) -> pd.DataFrame:
    sorted_means_df = means_df.sort_values(by=[("parameters", "Embedding dimensions"), ("parameters", "num_epochs")],
                                           ascending=False)
    return sorted_means_df.drop_duplicates(
        [("parameters", "Dataset"), ("parameters", "KG pool"), ("parameters", "Model")], keep="first")


def unify_parameters(means_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets rid of parameters that are different only across datasets. E.g.:
    KG-custom/finetuned/pretrained => custom
    Complex/Keci_RNN => model_RNN
    :param means_df: DataFrame
    :return: simplified_df: DataFrame
    """
    simplified_df = means_df.loc[:,
                    [("parameters", "Dataset"), ("parameters", "KG pool"), ("parameters", "Model"), ("Test", "Test_MRR")]]
    simplified_df[("parameters", "KG pool")] = simplified_df[("parameters", "KG pool")].replace(
        to_replace=r"^KG-.*$", value="Custom", regex=True)
    simplified_df[("parameters", "Model")] = simplified_df[("parameters", "Model")].replace(
        to_replace=["Keci", "ComplEx"], value="linear")
    simplified_df[("parameters", "Model")] = simplified_df[("parameters", "Model")].replace(
        to_replace=[r"^.*RNN", r"^.*GRU$", r".*LSTM$"], value=["RNN", "GRU", "LSTM"], regex=True)

    return simplified_df


def MRR_analysis_per_dataset(simplified_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns DF containing avg MRR, Rank, and improvement over original
    :param simplified_df:
    :return:
    """
    simplified_df.rename(columns={"MRR": "Avg MRR"}, inplace=True)
    simplified_df[("Test", "min ranks")] = simplified_df.groupby([("parameters", "Dataset")])[
        [("Test", "Avg MRR")]].rank(method="min", ascending=False)
    lin_df = simplified_df.loc[simplified_df[("parameters", "Model")] == "linear"]
    lin_df[("Test", "min ranks")] = lin_df.groupby([("parameters", "Dataset")])[[("Test", "min ranks")]].rank(
        method="min", ascending=False)
    diff_df = lin_df.sort_values([("parameters", "Dataset"), ("parameters", "KG pool")], ascending=False)
    diff_df[("Test", "Diff of avg MRR")] = diff_df.groupby([("parameters", "Dataset")])[
        [("Test", "Avg MRR")]].diff()
    diff_df[("Test", "Rel diff of avg MRR")] = diff_df.groupby([("parameters", "Dataset")])[
        [("Test", "Avg MRR")]].pct_change()
    return simplified_df, diff_df


def store_csvs(dataframes: list[pd.DataFrame], locs: list[str], names: list[str]) -> None:
    for i in range(len(dataframes)):
        dataframes[i].to_csv(os.path.join("Data analysis", "results", locs[i], "%s.csv" % names[i]),
                             index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--summarize_experiments', action='store_true')
args = parser.parse_args()

columns = [('parameters', 'Dataset'), ('parameters', 'KG pool'), ('parameters', 'Corpus input'),
           ('parameters', 'Tie breaker'), ('parameters', 'Vocab size'), ('parameters', 'Model'),
           ('parameters', 'Embedding dimensions'), ('parameters', 'Learning rate'),
           ('Val', 'H@1'), ('Val', 'H@3'), ('Val', 'H@10'), ('Val', 'MRR')]
if args.summarize_experiments:
    val_df = pd.read_pickle("Experiments/Summaries/sorted_df.pkl")
    val_df.sort_values(by=[("parameters", "Dataset"), ("Val", "MRR")], ascending=False, inplace=True)
    datasets = val_df[("parameters", "Dataset")].unique()

    for param in [["KG pool", "Corpus input"], ["Tie breaker"], ["Vocab size"]]:
        file_root = os.path.join("Data analysis", "results", "_".join(param))
        os.makedirs(file_root, exist_ok=True)
        df_parameter_ranks, df_dataset_avg_ranks = parameter_analysis(val_df, param)
        df_parameter_ranks.to_csv(os.path.join(file_root, "_min_ranks.csv"), index=True)
        df_dataset_avg_ranks.to_csv(os.path.join(file_root, "_avg_ranks.csv"), index=True)

linear_BytE_df = pd.read_pickle("mainBenchmarks/Summaries/sorted_df.pkl")
rnn_BytE_df = pd.read_pickle("test_benchmarks/Summaries/sorted_df.pkl")
multtrunc_df = pd.read_pickle("Multtrunc/Summaries/sorted_df.pkl")

linear_means_df = clean_benchmarks(create_means_from_summaries(linear_BytE_df))
rnn_means_df = create_means_from_summaries(rnn_BytE_df)
multtrunc_means_df = create_means_from_summaries(multtrunc_df)

combined_df = pd.concat([linear_BytE_df, rnn_BytE_df]).sort_values(
    by=[("parameters", "Dataset"), ("Test", "MRR")],
    ascending=False)
combined_means_df = pd.concat([linear_means_df, rnn_means_df]).sort_values(
    by=[("parameters", "Dataset"), ("Test", "MRR")], ascending=False)

# Save 6 sorted csv: 3 containing singular benchmarks, 3 containing means of benchmarks
os.makedirs(os.path.join("Data analysis", "results", "mainBenchmarks"), exist_ok=True)
os.makedirs(os.path.join("Data analysis", "results", "test_benchmarks"), exist_ok=True)
os.makedirs(os.path.join("Data analysis", "results", "benchmarks"), exist_ok=True)
os.makedirs(os.path.join("Data analysis", "results", "Multtrunc"), exist_ok=True)
linear_BytE_df.sort_values(by=[("parameters", "Dataset"), ("Test", "MRR")], ascending=False, inplace=True)
rnn_BytE_df.sort_values(by=[("parameters", "Dataset"), ("Test", "MRR")], ascending=False, inplace=True)
store_csvs([linear_BytE_df, linear_means_df, rnn_BytE_df, rnn_means_df, multtrunc_df, multtrunc_means_df,
            combined_df, combined_means_df],
           2 * ["mainBenchmarks"] + 2 * ["test_benchmarks"] + 2 * ["Multtrunc"] + 2 * ["benchmarks"],
           4 * ["Benchmarks", "Benchmark_means"])

simplified_df = unify_parameters(combined_means_df)
MRR_analyzed_df, lin_analyzed_df = MRR_analysis_per_dataset(simplified_df)
avg_MRR_Multtrunc = multtrunc_means_df.loc[:,
                    [("parameters", "Dataset"), ("parameters", "KG pool"), ("parameters", "Model"), ("Test", "MRR")]]
avg_MRR_Multtrunc[("Test", "min ranks")] = avg_MRR_Multtrunc.groupby([("parameters", "Dataset")])[
        [("Test", "MRR")]].rank(method="min", ascending=False)
store_csvs([MRR_analyzed_df, lin_analyzed_df, avg_MRR_Multtrunc],
           ["benchmarks", "mainBenchmarks", "Multtrunc"],
           3 * ["Average MRR (cleaned and ranked)"])

MRR_sum_df = MRR_analyzed_df[MRR_analyzed_df[("Test", "min ranks")] == 1.0].drop(
    columns=[("Test", "min ranks")])
MRR_sum_df = MRR_sum_df[[("parameters", "Model")]].value_counts(normalize=True).reset_index()
MRR_sum_df.rename(columns={"proportion": "First places in %"}, inplace=True)
linear_sum_df = lin_analyzed_df[lin_analyzed_df[("Test", "min ranks")] == 1.0].drop(
    columns=[("Test", "min ranks")])
linear_sum_df = linear_sum_df[[("parameters", "KG pool")]].value_counts(normalize=True).reset_index()
linear_sum_df.rename(columns={"proportion": "First places in %"}, inplace=True)
store_csvs([MRR_sum_df, linear_sum_df], ["benchmarks", "mainBenchmarks"],
           ["Overview of linearization methods", "Overview original vs custom"])

# truncation benchmarks are last
trunc_BytE_df = pd.read_pickle("truncmarks/Summaries/sorted_df.pkl")
trunc_means_df = create_means_from_summaries(trunc_BytE_df)
os.makedirs(os.path.join("Data analysis", "results", "truncmarks"), exist_ok=True)
trunc_BytE_df.sort_values(by=[("parameters", "Dataset"), ("Test", "MRR")], ascending=False, inplace=True)
store_csvs([trunc_BytE_df, trunc_means_df], 2 * ["truncmarks"], ["Benchmarks", "Benchmark_means"])

simp_trunc_df = unify_parameters(trunc_means_df)
simp_trunc_df.rename(columns={"MRR": "Avg MRR"}, inplace=True)
simp_trunc_df[("Test", "min ranks")] = simp_trunc_df.groupby([("parameters", "Dataset"), ("parameters", "KG pool")])[
    [("Test", "Avg MRR")]].rank(method="min", ascending=False)
best_trunc_df = simp_trunc_df[simp_trunc_df[("Test", "min ranks")] == 1.0].drop(columns=[("Test", "min ranks")])
prop_trunc_df = best_trunc_df[[("parameters", "Model")]].value_counts(normalize=True).reset_index()
prop_trunc_df.rename(columns={"proportion": "First places in %"}, inplace=True)
store_csvs([simp_trunc_df, best_trunc_df, prop_trunc_df], 3 * ["truncmarks"],
           ["MRR for each truncation", "Best truncation", "overview of best truncations"])
