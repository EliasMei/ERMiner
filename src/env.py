'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-20 07:38:04
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 15:28:22
'''
from doctest import master
import keyword
import time
import math
import logging
from random import random
import gym
import os
import numpy as np
from gym import spaces
from tree import Tree
from utils import CandidateCounter, merge_counts
from rule import RuleParser
from sklearn.preprocessing import OneHotEncoder
from utils import shrinkage_domain
from copy import deepcopy
import pandas as pd
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Env")

with open("../data/meta_data.json", "r") as f:
    meta_data = json.load(f)

def make_env(dataset="Location", stop_reward=0.01, k=50, maxd=200, supp=10, domain_path=None):
    """Build RLMiner Environment

    Args:
        dataset (str, optional): dataset name. Defaults to "Location".
        stop_reward (float, optional): reward value for the stop action. Defaults to 0.01.
        k (int, optional): topk rules set by users. Defaults to 50.
        maxd (int, optional): max domain size limit. Defaults to 200.
        supp (int, optional): support threshold. Defaults to 10.
        domain_path (_type_, optional): path of domain data . Defaults to None.

    Returns:
        _type_: _description_
    """
    x_disc_attrs = meta_data[dataset]["disc"]
    x_cont_attrs = meta_data[dataset]["cont"]
    y_attr = meta_data[dataset]["y_attr"]
    input_path = f"../data/{dataset}/input_data.csv"
    master_path = f"../data/{dataset}/master_data.csv"
    type_dict = {} 
    for col in x_disc_attrs+x_cont_attrs:
        if col in x_disc_attrs:
            type_dict[col] = "str"
        else:
            type_dict[col] = "float64"
    input_data = pd.read_csv(input_path, dtype=type_dict)
    master_data = pd.read_csv(master_path, dtype=type_dict)
    match = meta_data[dataset]["match"]
    if domain_path is not None:
        domain = pd.read_csv(domain_path, dtype=type_dict)
    else:
        domain = None
    env = ERMinerEnv(input_data, master_data, x_disc_attrs, x_cont_attrs, y_attr, match=match, stop_reward=stop_reward, k=k, maxd=maxd, supp_threshold=supp,domain=domain)
    return env

class ERMinerEnv(gym.Env):
    def __init__(self, input_data, master_data, x_disc_attrs, x_cont_attrs, y_attr, match, k=50, maxd=100, stop_reward=0.001, supp_threshold=0.01, domain=None) -> None:
        """Init Function of RLMiner

        Args:
            input_data (pd.DataFrame): input data 
            master_data (pd.DataFrame): master data
            x_disc_attrs (list): discrete attribute names
            x_cont_attrs (list): continuous attribute names
            y_attr (str): target attribute name
            match (dict): schema match
            k (int, optional): the number of topk. Defaults to 50.
            maxd (int, optional): maximum domain size. Defaults to 100.
            stop_reward (float, optional): reward for the stop action. Defaults to 0.001.
            supp_threshold (float, optional): support threshold. Defaults to 0.01.
            domain (pd.DataFrame, optional): data file store domains. Defaults to None.
        """
        self.input_data = input_data
        self.master_data = master_data
        self.labeled_data = input_data
        self.x_disc_attrs = x_disc_attrs
        self.x_cont_attrs = x_cont_attrs
        x_attrs = [col for col in x_disc_attrs+x_cont_attrs if col in input_data.columns]
        self.x_attrs = x_attrs
        self.y_attr = y_attr
        self.match = match
        self.stop_reward = stop_reward
        self.k = k
        self.supp_threshold = supp_threshold
        encoders = dict()
        self.domain = input_data if domain is None else domain
        # use shrink domain as preprocessing
        for col in x_attrs:
            is_continuous = True if col in x_cont_attrs else False
            tmp = shrinkage_domain(self.domain[col], k=maxd, continuous=is_continuous)
            domain = tmp.unique().reshape(-1,1)
            encoder = OneHotEncoder()
            encoder.fit(domain)
            encoders[col] = encoder
        self.parser = RuleParser(encoders, match, x_attrs)
        self.tree = Tree(x_attrs=x_attrs, y_attr=y_attr, rule_parser=self.parser)
        self.enc_dim = self.parser.get_enc_dim()
        self.observation_space = spaces.Discrete(self.enc_dim)
        self.action_space = spaces.Discrete(self.enc_dim+1)

        self.counter = CandidateCounter(x_disc_attrs, x_cont_attrs)
        self.reward_dict = dict() # store (rule:reward)
        self.score_dict = dict() # store (rule:score)
        self.input_space = dict() # store (rule:input_ixs)
        self.master_space = dict() # store (rule:master_ixs)
        self.stop_rules = set()

        self.steps = 0

    def reset(self):
        """reset the environment

        Returns:
            dict: initialized observation and mask
        """
        self.state = np.zeros(self.enc_dim, dtype=int)
        self.curr_rule = None
        self.tree = Tree(x_attrs=self.x_attrs, y_attr=self.y_attr, rule_parser=self.parser)
        self.steps = 0
        self.mask = self.parser.get_mask_from_state(self.state)
        self.mask[-1] = 0
        return {"obs":self.state, "mask":self.mask}
        
    def step(self, action):
        """update the environment according to the action and return the reward

        Args:
            action (int): action index

        Returns:
            dict: observation, mask, reward information and info
        """
        self.steps += 1
        # if the action is "stop", then RLMiner will move to the next node.
        if action == self.enc_dim:
            reward = self.stop_reward
            next_node = self.tree.get_next_node()
        else:
            encoding = deepcopy(self.state)
            encoding[action] = 1
            rule = self.parser.encoding_to_rule(encoding)
            # if the rule is explored, the reuse the reward
            if rule in self.reward_dict:
                reward = self.reward_dict[rule]
                score = self.score_dict[rule]
                stop_flag = True if rule in self.stop_rules else False
            # otherwise, evaluate the discovered rule, calculate and store the reward
            else:
                if self.curr_rule is None:
                    score, input_ixs, master_ixs = self.eval_rule(rule)
                else:
                    score, input_ixs, master_ixs = self.eval_rule(self.curr_rule, action)
                self.input_space[rule] = input_ixs
                self.master_space[rule] = master_ixs
                if score["support"] >= self.supp_threshold and score["uncertainty"] < 1.:
                    stop_flag = False 
                else:
                    stop_flag = True
                reward = self.cal_reward(score)
                self.reward_dict[rule] = reward
                self.score_dict[rule] = score
                if stop_flag:
                    self.stop_rules.add(rule)
            # encourage / penalize the RLMiner if discover rules from the node with low / high utility 
            valid = True if score["support"] >= self.supp_threshold else False
            if valid:
                if self.curr_rule in self.tree.leaves:
                    reward = reward + (reward-self.reward_dict.get(self.curr_rule,0))
            next_node = self.tree.update(encoding, stop_flag, valid)
        
        # check whether the RLMiner ends
        if next_node and len(self.tree.get_leaves()) < self.k and self.steps < 300:
            self.state = next_node.get_state()
            self.curr_rule = self.parser.encoding_to_rule(self.state)
            self.mask = self.tree.get_action_mask(next_node)
            done = False
        else:
            self.state = np.ones(self.enc_dim, dtype=int)
            self.curr_rule = None
            self.mask = np.zeros(self.enc_dim+1)
            done = True
        if done:
            print(f"Steps: {self.steps}. RuleNum: {len(self.tree.get_leaves())}.") 
        return {"obs":self.state, "mask":self.mask}, reward, done, {}

    def rule_set_query(self):
        """repair with a set of eRs

        Args:
            rules (list): a list of editing rules

        Returns:
            dict: candidate fixes for all tuples
        """
        all_counts = dict()
        for rule in self.tree.get_leaves():
            rule_counts = self.counter.counts(self.input_data, self.master_data, self.y_attr, rule)
            all_counts = merge_counts(all_counts, rule_counts)
        return all_counts

    def topk_rules(self, k):
        """return the top-k eRs

        Args:
            k (int): number of k

        Returns:
            list: top-k rules with the highest utility
        """
        rule_scores = {rule:self.score_dict[rule] for rule in self.tree.get_leaves()}
        sorted_rules = sorted(rule_scores.items(), key=lambda x:x[1]["utility"], reverse=True)
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

    def eval_rule(self, rule, action_ix=None):
        """Evaluate the measures of the given rule and return the subspace for acceleration

        Args:
            rule (EditingRule): to-evaluate editing rule
            action_ix (int, optional): the index of the chosen action

        Returns:
            dict, pd.Index, pd.Index: measures, input subspace index, master subspace index
        """
        if sum(self.state) == 0:
            cand_counts = self.counter.counts(self.input_data, self.master_data, self.y_attr, rule, self.supp_threshold)
            input_ixs, master_ixs = self.input_data.index, self.master_data.index
        else:
            ori_input_ixs, ori_master_ixs = self.input_space[self.curr_rule], self.master_space[self.curr_rule]
            action, action_type = self.parser.ix_cond_dict[action_ix]
            cand_counts, input_ixs, master_ixs = self.counter.counts_with_sample(self.input_data.loc[ori_input_ixs], self.master_data.loc[ori_master_ixs], self.y_attr, self.curr_rule, action, action_type, self.supp_threshold)
        s_score = self.cal_support(cand_counts)
        u_score = self.cal_certainty(cand_counts)
        q_score = self.cal_quality(cand_counts)
        utility = math.log(max(s_score,1.1),10)**2*(u_score+q_score)
        score = {"support":s_score,"uncertainty":u_score,"quality":q_score, "utility":utility}
        return score, input_ixs, master_ixs

    def cal_support(self, cand_counts):
        """calculate the support

        Args:
            cand_counts (dict): candidate fixes 

        Returns:
            int: support
        """
        return len(cand_counts)

    def cal_certainty(self, cand_counts):
        """calculate the certainty

        Args:
            cand_counts (dict): candidate fixes 

        Returns:
            _type_: certainty
        """
        all_ratio = list()
        for ix, cnt_dict in cand_counts.items():
            all_ratio.append(max(cnt_dict.values()))
        if all_ratio:
            return sum(all_ratio) / len(all_ratio)
        return 0

    def cal_quality(self, cand_counts):
        """calculate the quality

        Args:
            cand_counts (dict): candidate fixes 

        Returns:
            _type_: quality
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

    def cal_reward(self, score):
        """calculate the reward according to rule measures

        Args:
            score (dict): rule measures

        Returns:
            float: reward
        """
        if score["support"] < self.supp_threshold:
            return -0.01
        return score["utility"]
