'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-20 07:38:14
LastEditors: Yinan Mei
LastEditTime: 2022-04-26 07:16:37
'''
from collections import Counter
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
import os
import torch
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Utils")

def prefix_by_sep(data, k):
    data = data.str.replace("-", " ")
    data = data.str.replace("_", " ")
    substr_list = []
    for v in data:
        sub_strings = str(v).split()
        for ix, sub_str in enumerate(sub_strings):
            if ix < len(substr_list):
                substr_list[ix].append(" ".join(sub_strings[:ix+1]))
            else:
                substr_list.append([])
                substr_list[ix].append(" ".join(sub_strings[:ix+1]))
    prefix_ix = 0
    for ix, l in enumerate(substr_list):
        if len(Counter(l)) < k:
            prefix_ix = ix
            continue
        break
    def replace_func(v):
        sub_strings = str(v).split()
        return " ".join(sub_strings[:prefix_ix+1])
    data = data.apply(replace_func)
    return data

def prefix_by_str(data, k):
    data = data.astype(str)
    max_len = int(max(data.str.len()))
    prefix_pos = 0
    for i in range(1, max_len+1):
        if len(data.str[:i].unique()) <= k:
            prefix_pos = i 
            continue
        break
    def replace_func(v):
        if v == "nan": return v
        return v[:prefix_pos]
    data = data.apply(replace_func)
    return data

def topk_frequency_encoding(data, k):
    counts = data.value_counts()
    topk_classes = counts[:k].index
    data = data.apply(lambda x:x if x in topk_classes else "Other")
    return data

def check_specials(data, threshold=0.9):
    specials = [" ", "-", "_"]
    flag = False
    for c in specials:
        cnt = 0
        for v in data:
            cnt += (c in str(v))
        if (cnt / len(data)) > threshold:
            flag = True
            break
    return flag

def shrinkage_domain(data:pd.Series, k:int, threshold:float=0.9, continuous=False):
    counts = data.value_counts()
    if len(counts) < k:
        return data
    # if (sum(counts[:k]) / len(data)) >= threshold:
    #     data = topk_frequency_encoding(data, k)
    #     prefix_pos = None
    #* For simplicity of querying, string prefix is more suitable
    # flag = check_specials(data, threshold)
    # if flag:
    #     data = prefix_by_sep(data, k)
    else:
        # data, prefix_pos = prefix_by_str(data, k)
        if continuous:
            data = pd.qcut(data, q=50, duplicates="drop").astype(str)
        else:
            data = prefix_by_str(data, k)
    return data

def merge_counts(cand_counts, new_cand_counts):
    for ix, new_counts in new_cand_counts.items():
        if ix not in cand_counts:
            cand_counts[ix] = new_counts
            continue
        ori_counts = deepcopy(cand_counts[ix])
        for v, cnt in new_counts.items():
            if v in ori_counts:
                ori_counts[v] += cnt
            else:
                ori_counts[v] = cnt
        cand_counts[ix] = ori_counts
    return cand_counts

class CandidateCounter(object):
    def __init__(self, x_disc_attrs, x_cont_attrs) -> None:
        super().__init__()
        self.x_disc_attrs = x_disc_attrs
        self.x_cont_attrs = x_cont_attrs
    
    def count_candidates(self, input_df, master_df, y_attr):
        cand_counts = master_df[y_attr].value_counts().to_dict()
        #* transform count to ratio. Helping ID-like attribute dominates 
        cnt_sum = sum([cnt for v, cnt in cand_counts.items()])
        cand_counts = {v:cnt/cnt_sum for v, cnt in cand_counts.items()}
        if cand_counts:
            count_dict = {ix:cand_counts for ix in input_df.index}
        else:
            count_dict = {}
        return count_dict

    def enumerate_wildcards(self, input_df, master_df, y_attr, attrs, attrs_m):
        cand_counts = dict()
        cand_conds = input_df[attrs].value_counts().index
        cond_ixs_dict = input_df.groupby(attrs).groups
        if len(attrs) == 1:
            master_cond_ixs = pd.MultiIndex.from_arrays([master_df.set_index(attrs_m).index])
        else:
            master_cond_ixs = master_df.set_index(attrs_m).index
        tmp_df = master_df[master_cond_ixs.isin(cand_conds)]
        master_counts = tmp_df.pivot_table(index=attrs_m, columns=y_attr, aggfunc='size', fill_value=0)
        if len(attrs) == 1:
            cand_conds = [cond[0] for cond in cand_conds]
        for cond in cand_conds:
            input_ixs = cond_ixs_dict[cond]
            if cond not in master_counts.index:
                continue
            counts = master_counts.loc[cond]
            #* transform count to ratio. Helping ID-like attribute dominates 
            cnt_sum = sum([cnt for v, cnt in counts.items()])
            counts_ratio = {v:cnt/cnt_sum for v, cnt in counts.items()}
            tmp_cand_counts = {ix:counts_ratio for ix in input_ixs}
            # cand_counts = merge_counts(cand_counts, tmp_cand_counts)
            cand_counts.update(tmp_cand_counts)
        return cand_counts

    def query(self, data, attr, value, continuous):
        value = str(value)
        if attr is None:
            return None
        if continuous:
            if "," not in value:
                cond = data[attr].astype(str) == value
            else:
                ix = value.index(",")
                left, right = float(value[1:ix]), float(value[ix+2:-1])
                cond = data[attr].astype(float).between(left, right)
        else:
            cond = data[attr].astype(str).str.startswith(value)
        return cond

    def counts(self, input_data, master_data, y_attr, rule, supp=1):
        input_cond, master_cond = None, None
        for attr, value in rule.pattern.items():
            is_continuous = True if attr in self.x_cont_attrs else False
            input_new_cond = self.query(input_data, attr, value, is_continuous)
            if input_cond is not None:
                input_cond = input_cond & input_new_cond
            else:
                input_cond = input_new_cond
            attr_m = rule.lhs_attrs.get(attr, None)
            master_new_cond = self.query(master_data, attr_m, value, is_continuous)
            if master_new_cond is not None:
                if master_cond is not None:
                    master_cond = master_cond & master_new_cond
                else:
                    master_cond = master_new_cond
        covered_input = input_data[input_cond] if input_cond is not None else input_data
        covered_master = master_data[master_cond] if master_cond is not None else master_data
        if len(covered_input) < supp:
            # return {}, covered_input.index, covered_master.index
            return {}
        if rule.wildcard_attrs:
            cand_counts = self.enumerate_wildcards(covered_input, covered_master, y_attr, rule.wildcard_attrs, rule.wildcard_attrs_m)
        else:
            cand_counts = self.count_candidates(covered_input, covered_master, y_attr)
        return cand_counts

    def counts_with_sample(self, input_data, master_data, y_attr, rule, action, action_type="lhs", supp=1):
        wildcard_attrs = deepcopy(rule.wildcard_attrs)
        if action_type == "lhs":
            covered_input = input_data
            covered_master = master_data
            attr, attr_m = action
            # if attr is already considered as pattern, then we will add it into wildcard attributes
            if attr in rule.pattern:
                wildcard_attrs, wildcard_attrs_m = rule.wildcard_attrs, rule.wildcard_attrs_m
            else:
                wildcard_attrs, wildcard_attrs_m = [action[0]], [action[1]]
                wildcard_attrs.extend(rule.wildcard_attrs)
                wildcard_attrs_m.extend(rule.wildcard_attrs_m)
        else:
            attr, value = action[0], str(action[1])
            attr_m = rule.lhs_attrs.get(attr, None)
            is_continuous = True if attr in self.x_cont_attrs else False
            input_cond = self.query(input_data, attr, value, continuous=is_continuous)
            master_cond = self.query(master_data, attr_m, value, continuous=is_continuous)
            if attr_m is not None and sum(master_cond)==0:
                return dict(), [], []
            covered_input = input_data[input_cond]
            covered_master = master_data[master_cond] if master_cond is not None else master_data
            # if attr is already considered as wildcard attrs, then we should remove it
            wildcard_attrs, wildcard_attrs_m = deepcopy(rule.wildcard_attrs), deepcopy(rule.wildcard_attrs_m)
            if attr in wildcard_attrs:
                wildcard_attrs.remove(attr)
                wildcard_attrs_m.remove(attr_m)
        
        if len(covered_input) < supp:
            return {}, covered_input.index, covered_master.index

        if wildcard_attrs:
            cand_counts = self.enumerate_wildcards(covered_input, covered_master, y_attr, wildcard_attrs, wildcard_attrs_m)
        else:
            cand_counts = self.count_candidates(covered_input, covered_master, y_attr)
        return cand_counts, covered_input.index, covered_master.index

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True