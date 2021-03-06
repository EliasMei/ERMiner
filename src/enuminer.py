'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-29 13:35:38
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 15:20:51
'''

import argparse
from collections import defaultdict
import pickle
import time
import logging
import pandas as pd 
from copy import deepcopy
from utils import merge_counts, CandidateCounter
from utils import shrinkage_domain
import json
import os
import math
from rule import EditingRule
from tqdm import tqdm 

import argparse
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Baseline")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HOSP')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--maxd', type=int, default=200)
    parser.add_argument('--supp', type=float, default=0.01)
    return parser.parse_args()

class EnuMiner(object):
    def __init__(self, input_domains, master_data, y_attr, match, x_disc_attrs, x_cont_attrs, input_data, supp_threshold=10) -> None:
        """Init Function

        Args:
            input_domains (pd.DataFrame): input domains
            master_data (pd.DataFrame): clean master data
            y_attr (str): target attribute name
            match (dict): schema match results
            x_disc_attrs (list): discrete attributes
            x_cont_attrs (list): continuous attributes
            input_data (pd.DataFrame, optional): input data.
            supp_threshold (int, optional): support threshold. Defaults to 10.
        """
        self.input_domains = input_domains
        self.x_attrs = [col for col in x_disc_attrs+x_cont_attrs if col in input_domains.columns]
        self.y_attr = y_attr
        self.input_data = input_data
        self.labeled_data = input_data
        self.master_data = master_data
        self.x_disc_attrs = x_disc_attrs
        self.x_cont_attrs = x_cont_attrs
        self.match = match
        self.counter = CandidateCounter(x_disc_attrs, x_cont_attrs)
        self.input_space, self.master_space = {}, {}
        self.supp_threshold = supp_threshold

    def cal_quality(self, cand_counts):
        """Evaluate the rule quality according to the returned fixes by it

        Args:
            cand_counts (dict): returned fixes by the rule

        Returns:
            float: quality score
        """
        quality, cnt = 0, 0
        for ix in self.labeled_data.index:
            if ix in cand_counts:
                cnt += 1
                major_cls = sorted(cand_counts[ix].items(), key=lambda x:x[1], reverse=True)[0][0]
                q = 1 if major_cls == self.labeled_data.loc[ix, self.y_attr] else -1
                quality += q
        if cnt == 0:
            return 0
        return quality / cnt

    def eval_rule(self, rule, action=None, action_type=None):
        """Evaluate the measures of the given rule and return the subspace for acceleration

        Args:
            rule (EditingRule): to-evaluate editing rule
            action (tuple, optional): the chosen action that generates the rule. Defaults to None.
            action_type (str, optional): type of the actions. Defaults to None.

        Returns:
            dict, pd.Index, pd.Index: measures, input subspace index, master subspace index
        """
        s_score, u_score, q_score, utility = 0, 0, 0, 0
        if self.input_data is not None:
            if action:
                ori_input_ixs, ori_master_ixs = self.input_space[rule], self.master_space[rule]
                cand_counts, input_ixs, master_ixs = self.counter.counts_with_sample(self.input_data.loc[ori_input_ixs], self.master_data.loc[ori_master_ixs], self.y_attr, rule, action, action_type, self.supp_threshold)
            else:
                cand_counts = self.counter.counts(self.input_data, self.master_data, self.y_attr, rule, self.supp_threshold)
                input_ixs, master_ixs = self.input_data.index, self.master_data.index
            if len(cand_counts) > 0:
                s_score = len(cand_counts)
                all_ratio = []
                for ix, cnt_dict in cand_counts.items():
                    all_ratio.append(max(cnt_dict.values()))
                u_score = sum(all_ratio) / len(all_ratio)
                q_score = self.cal_quality(cand_counts)
            utility = math.log(max(s_score,1.1),10)**2*(u_score+q_score)

        return {"support":s_score,"uncertainty":u_score,"quality":q_score, "utility":utility}, input_ixs, master_ixs

    def mine(self):
        """discover editing rules

        Returns:
            list: all valid rules
        """
        self.rule_dict, cand_set = dict(), set()
        self.valid_rules = dict()
        loop = 0
        # initialization
        for attr in self.x_attrs:
            for a_m in self.match.get(attr, []):
                rule = EditingRule({attr:a_m},dict(),self.x_attrs)
                score, input_ixs, master_ixs = self.eval_rule(rule)
                self.rule_dict[rule] = score
                self.input_space[rule] = input_ixs
                self.master_space[rule] = master_ixs
                if score["uncertainty"] < 1.:
                    cand_set.add(rule)
        logger.info(f"Loop {loop}: Initialized. [Candidate Size]={len(cand_set)}. [Rule Num]={len(self.rule_dict)}")
        # Enumeration
        while cand_set:
            tmp_set = set()
            for rule in tqdm(cand_set):
                ori_input_ixs, ori_master_ixs = self.input_space[rule], self.master_space[rule]
                lhs_attrs, pattern = rule.lhs_attrs, rule.pattern
                for attr in self.x_attrs:
                    # enumerate attribute pairs for LHS
                    if attr in rule.lhs_attrs:
                        continue
                    for a_m in self.match.get(attr, []):
                        new_lhs_attrs = deepcopy(lhs_attrs)
                        new_lhs_attrs[attr] = a_m
                        new_rule = EditingRule(new_lhs_attrs,pattern,self.x_attrs)
                        score, input_ixs, master_ixs = self.eval_rule(rule)
                        self.rule_dict[new_rule] = score
                        self.input_space[new_rule] = input_ixs
                        self.master_space[new_rule] = master_ixs
                        if score["uncertainty"] < 1:
                            tmp_set.add(new_rule)
                        if score["support"]>=self.supp_threshold:
                            self.valid_rules[new_rule] = score

                    # enumerate attribute value pairs for pattern
                    if attr not in pattern:
                        for v in self.input_domains.loc[ori_input_ixs][attr].unique():
                            new_pattern = deepcopy(pattern)
                            new_pattern[attr] = v
                            new_rule = EditingRule(lhs_attrs,new_pattern,self.x_attrs)
                            score, input_ixs, master_ixs = self.eval_rule(rule)
                            self.rule_dict[new_rule] = score
                            self.input_space[new_rule] = input_ixs
                            self.master_space[new_rule] = master_ixs
                            if 0<score["uncertainty"] < 1 and score["support"]>=self.supp_threshold:
                                tmp_set.add(new_rule)
                            if score["support"]>=self.supp_threshold:
                                self.valid_rules[new_rule] = score
            cand_set = tmp_set
            loop += 1
            logger.info(f"Loop {loop}: Done. [Candidate Size]={len(cand_set)}. [Rule Num]={len(self.rule_dict)}. [Valid Rules]={len(self.valid_rules)}")
        return self.valid_rules

    def top_k(self, k):
        """return the top-k eRs

        Args:
            k (int): number of k

        Returns:
            list: top-k rules with the highest utility
        """
        sorted_rules = sorted(self.valid_rules.items(), key=lambda x:x[1]["utility"], reverse=True)
        topk_rules = []
        for rule, _ in sorted_rules:
            flag = True
            for former_rule in topk_rules:
                if former_rule.dominate(rule):
                    flag = False
                    break
            if flag:
                topk_rules.append(rule)
            if len(topk_rules) == k:
                break
        self.topk_rules = topk_rules
        return topk_rules

    def rule_set_query(self, rules):
        """repair with a set of eRs

        Args:
            rules (list): a list of editing rules

        Returns:
            dict: candidate fixes for all tuples
        """
        all_counts = dict()
        for rule in rules:
            rule_counts = self.counter.counts(self.input_data, self.master_data, self.y_attr, rule)
            all_counts = merge_counts(all_counts, rule_counts)
        return all_counts     

if __name__ == "__main__":
    with open("../data/meta_data.json", "r") as f:
        meta_data = json.load(f)
    args = get_args()
    
    # load meta-data
    x_disc_attrs = meta_data[args.dataset]["disc"]
    x_cont_attrs = meta_data[args.dataset]["cont"]
    y_attr = meta_data[args.dataset]["y_attr"]
    type_dict = {y_attr:str} 
    for col in x_disc_attrs+x_cont_attrs:
        if col in x_disc_attrs:
            type_dict[col] = "str"
        else:
            type_dict[col] = "float64"
    
    # load data
    input_path = f"../data/{args.dataset}/input_data.csv"
    master_path = f"../data/{args.dataset}/master_data.csv"
    input_data = pd.read_csv(input_path, dtype=type_dict)
    master_data = pd.read_csv(master_path, dtype=type_dict)
    x_attrs = [col for col in x_disc_attrs+x_cont_attrs if col in input_data.columns]

    # compress domains 
    input_domains = deepcopy(input_data)
    for col in x_attrs:
        is_continuous = True if col in x_cont_attrs else False
        tmp = shrinkage_domain(input_data[col], k=args.maxd, continuous=is_continuous)
        input_domains[col] = tmp
    match = meta_data[args.dataset]["match"]
    
    # start mining 
    start_time = time.perf_counter()
    miner = EnuMiner(input_domains, master_data, y_attr, match=match, x_disc_attrs=x_disc_attrs, x_cont_attrs=x_cont_attrs, input_data=input_data, supp_threshold=args.supp)
    miner.mine()
    rules = miner.top_k(args.k)
    logging.info(f"Rule Ranking Done.")
    print("Rule Num:",len(miner.rule_dict))
    end_time = time.perf_counter()
    
    # save results 
    if not os.path.exists(f"../output/{args.dataset}/"):
        os.makedirs(f"../output/{args.dataset}/")
    if not os.path.exists(f"../tmp/{args.dataset}/"):
        os.makedirs(f"../tmp/{args.dataset}/")
    
    # save mined rules
    with open(f"../tmp/{args.dataset}/EnuMiner-rules.pkl", "wb") as f:
        pickle.dump(rules, f)
    
    # save candidate fixes
    cand_counts = miner.rule_set_query(rules)
    with open(f"../tmp/{args.dataset}/EnuMiner-cand_counts.pkl", "wb") as f:
        pickle.dump(cand_counts, f)
    logging.info(f"Time: {end_time-start_time}")

    # save time cost
    time_logger_path = f"../output/{args.dataset}/time_logger.json"
    if os.path.exists(time_logger_path):
        with open(time_logger_path, "r") as f:
            time_dict = json.load(f)
    else:
        time_dict = dict()
    if str(args.num) not in time_dict:
        time_dict[str(args.num)] = dict()
    if "EnuMiner" not in time_dict[str(args.num)]:
        time_dict[str(args.num)]["EnuMiner"] = dict()
    time_dict[str(args.num)]["EnuMiner"][str(args.seed)] = end_time-start_time
    with open(time_logger_path, "w") as f:
        json.dump(time_dict, f)
