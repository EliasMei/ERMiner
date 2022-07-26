'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-24 12:41:48
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 14:36:35
'''
import time
import json
import os
import argparse
import pickle

import pandas as pd
import numpy as np
import torch

from env import  make_env
from net import Rainbow
from tianshou.policy import DQNPolicy, RainbowPolicy
from tianshou.data import Batch
from tianshou.utils.net.common import Net

from utils import seed_everything


def get_args():
    parser = argparse.ArgumentParser()
    # the parameters are found by Optuna
    parser.add_argument('--dataset', type=str, default='ERMiner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stopreward', type=float, default=0.01)
    parser.add_argument('--supp', type=float, default=10)
    parser.add_argument('--maxd', type=int, default=100)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--eps-test', type=float, default=0.01)
    parser.add_argument('--eps-train', type=float, default=0.73)
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.0625)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument(
        '--dueling-q-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument(
        '--dueling-v-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument('--algorithm', type=str, default='dqn')
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--domain_path', type=str, default=None)
    return parser.parse_args()

def discover(args):
    """discover a set of editing rules

    Args:
        args (Argparse): argument dict

    Raises:
        NotImplemented: _description_
    """
    # init RLMiner environment
    env = make_env(args.dataset, stop_reward=args.stopreward, k=args.k, maxd=args.maxd, supp=args.supp, domain_path=args.domain_path)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # model
    Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
    if args.algorithm == "dqn":
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            dueling_param=(Q_param, V_param),
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )
    elif args.algorithm == "rainbow":
        net = Rainbow(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            dueling_param=(Q_param, V_param),
            num_atoms=51,
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = RainbowPolicy(
            net,
            optim,
            args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)
    else:
        raise NotImplemented(f"Algorithm {args.algorithm} is not supported yet.")
    # load trained value network
    policy.load_state_dict(torch.load(f'./log/{args.dataset}/{args.num}/{args.algorithm}/policy.pth'))
    policy.eval()
    policy.set_eps(0.05)

    # reset environment
    obs = env.reset()
    obs = {k:np.array([v]) for k, v in obs.items()}
    policy.eval()
    # Start mining
    while True:
        action = policy.forward(Batch(obs=obs, info=None)).act[0]
        obs, reward, done, info = env.step(action)
        obs = {k:np.array([v]) for k, v in obs.items()}
        if done or len(env.tree.get_leaves()) > 50:
            break
    cand_counts = env.rule_set_query()
    # return the discovered valid rules
    rules = env.topk_rules(args.k)
    alg = args.algorithm if args.algorithm != "dqn" else ""
    # Save discovered
    with open(f"../tmp/{args.dataset}/ERMiner{alg}-rules.pkl", "wb") as f:
        pickle.dump(rules, f)
    # Save possible repairs
    with open(f"../tmp/{args.dataset}/ERMiner{alg}-cand_counts.pkl", "wb") as f:
        pickle.dump(cand_counts, f)
    print("Rule Num:",len(rules))
    return 

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(f"../output/{args.dataset}/"):
        os.makedirs(f"../output/{args.dataset}/")
    if not os.path.exists(f"../tmp/{args.dataset}/"):
        os.makedirs(f"../tmp/{args.dataset}/")

    seed_everything(args.seed)
    start_time = time.perf_counter()
    discover(args)
    end_time = time.perf_counter()
    print(f"Discovery Time[{args.seed}]: ", end_time-start_time)

    # Save time cost
    time_logger_path = f"../output/{args.dataset}/time_logger.json"
    if os.path.exists(time_logger_path):
        with open(time_logger_path, "r") as f:
            time_dict = json.load(f)
    else:
        time_dict = dict()
    if str(args.num) not in time_dict:
        time_dict[str(args.num)] = dict()
    alg = args.algorithm if args.algorithm != "dqn" else ""
    if f"ERMiner{alg}-discover" not in time_dict[str(args.num)]:
        time_dict[str(args.num)][f"ERMiner{alg}-discover"] = dict()
    time_dict[str(args.num)][f"ERMiner{alg}-discover"][str(args.seed)] = end_time-start_time
    with open(time_logger_path, "w") as f:
        json.dump(time_dict, f)