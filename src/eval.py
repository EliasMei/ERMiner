'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-24 12:41:58
LastEditors: Yinan Mei
LastEditTime: 2022-04-23 05:13:48
'''

import argparse
import pickle 
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import json

with open("../data/meta_data.json", "r") as f:
    meta_data = json.load(f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Covid')
    parser.add_argument('--num', type=int, default=2500)
    parser.add_argument('--method', type=str, default="ERMiner")
    parser.add_argument('--algorithm', type=str, default="dqn")
    return parser.parse_args()

def eval(dataset, num, method, seed_list=[0,1,2,3,4]):
    y_attr = meta_data[dataset]["y_attr"]
    total_p_score_dict, total_r_score_dict, total_f_score_dict = dict(), dict(), dict()
    total_avg_score_dict = dict()
    x_disc_attrs = meta_data[dataset]["disc"]
    x_cont_attrs = meta_data[dataset]["cont"]
    type_dict = {y_attr:"str"} 
    for col in x_disc_attrs+x_cont_attrs:
        if col in x_disc_attrs:
            type_dict[col] = "str"
        else:
            type_dict[col] = "float64"
    for i, seed in enumerate(seed_list):
        input_path = f"../data/{dataset}/input_data.csv"
        truth_path = f"../data/{dataset}/input_data_clean.csv"
        df = pd.read_csv(input_path, dtype=type_dict)
        truth = pd.read_csv(truth_path, dtype=type_dict)
        labels = truth[y_attr].unique()

        with open(f"../tmp/{dataset}/{method}-cand_counts.pkl", "rb") as f:
            cand_counts = pickle.load(f)
        y_pred, y_true = [], []
        for t in df.itertuples():
            ix = t.Index
            if ix not in cand_counts:
                v = "nan"
            else:
                v = sorted(cand_counts[ix].items(), key=lambda x:x[1], reverse=True)[0][0]
            y_pred.append(v)
            y_true.append(truth.loc[ix, y_attr])
        p_score = precision_score(y_true, y_pred, labels=labels,average=None)
        r_score = recall_score(y_true, y_pred, labels=labels, average=None)
        f_score = f1_score(y_true, y_pred, labels=labels, average=None)
        total_p_score_dict[i] = {label:p_score[ix] for ix, label in enumerate(labels)}
        total_r_score_dict[i] = {label:r_score[ix] for ix, label in enumerate(labels)}
        total_f_score_dict[i] = {label:f_score[ix] for ix, label in enumerate(labels)}
        avg_score_dict = {}
        for score_method in ["weighted"]:
            p = precision_score(y_true, y_pred, average=score_method)
            r = recall_score(y_true, y_pred, average=score_method)
            f1 = f1_score(y_true, y_pred, average=score_method)
            avg_score_dict[f"precision-{score_method}"] = p
            avg_score_dict[f"recall-{score_method}"] = r
            avg_score_dict[f"f1-{score_method}"] = f1
        total_avg_score_dict[i] = avg_score_dict
    avg_score_df = pd.DataFrame(total_avg_score_dict)
    avg_score_df["mean"] = avg_score_df.mean(axis=1)
    avg_score_df.to_csv(f"../output/{dataset}/{method}-avg_score.csv")
    print(avg_score_df)
    pd.DataFrame(total_p_score_dict).to_csv(f"../output/{dataset}/{method}-precision.csv")
    pd.DataFrame(total_r_score_dict).to_csv(f"../output/{dataset}/{method}-recall.csv")
    pd.DataFrame(total_f_score_dict).to_csv(f"../output/{dataset}/{method}-f1.csv")

if __name__ == "__main__":
    args = get_args()
    seed_list = range(5)
    if args.method == "ERMiner":
        if args.algorithm != "dqn":
            method = args.method+args.algorithm
        else:
            method = args.method
    else:
        method = args.method
    eval(dataset=args.dataset, num=args.num, method=method, seed_list=seed_list)