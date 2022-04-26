'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-24 12:41:35
LastEditors: Yinan Mei
LastEditTime: 2022-04-26 07:21:30
'''
from email.policy import default
import time
import os
import pprint
import argparse
from turtle import update

import pandas as pd
import json
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import ERMinerEnv, make_env
from net import Rainbow
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, C51Policy, RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from utils import seed_everything

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ERMiner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--decay', type=float, default=0.0002)
    parser.add_argument('--stopreward', type=float, default=0.1)
    parser.add_argument('--supp', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--maxd', type=int, default=100)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--eps-test', type=float, default=0.01)
    parser.add_argument('--eps-train', type=float, default=0.73)
    parser.add_argument('--buffer-size', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--update-per-step', type=float, default=0.0625)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument(
        '--dueling-q-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument(
        '--dueling-v-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument(
        '--algorithm', type=str, default="dqn")
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--expseed', type=int, default=-1)
    parser.add_argument('--pt_path', type=str, default=None)
    parser.add_argument('--domain_path', type=str, default=None)
    return parser.parse_args()

    
def train_dqn(args=get_args()):
    env = make_env(args.dataset, num=args.num, seed=args.seed, stop_reward=args.stopreward, k=args.k, maxd=args.maxd, alpha=args.alpha, supp=args.supp, domain_path=args.domain_path)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Stop Reward:", args.stopreward)
    # make environments
    train_envs = DummyVectorEnv(
        [lambda: make_env(args.dataset, num=args.num, seed=args.seed, stop_reward=args.stopreward, k=args.k, maxd=args.maxd, alpha=args.alpha, supp=args.supp, domain_path=args.domain_path) for _ in range(args.training_num)]
    )
    train_envs.seed(args.seed)
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
    
    # incremental exp
    if args.pt_path is not None:
        policy.load_state_dict(torch.load(args.pt_path))
    
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.dataset, str(args.num), args.algorithm)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer,train_interval=1,update_interval=1)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, f'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train * (1 - args.decay)**env_step, args.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,# test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_fn=save_fn,
        logger=logger
    )

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(f"../output/{args.dataset}/"):
        os.makedirs(f"../output/{args.dataset}/")
    if not os.path.exists(f"../tmp/{args.dataset}/"):
        os.makedirs(f"../tmp/{args.dataset}/")

    seed_everything(args.seed)
    start_time = time.perf_counter()
    train_dqn(args)
    end_time = time.perf_counter()
    print(f"Training Time[{args.seed}]: ", end_time-start_time)
    
    if not os.path.exists(f"../output/{args.dataset}/"):
        os.makedirs(f"../output/{args.dataset}/")

    time_logger_path = f"../output/{args.dataset}/time_logger.json"
    if os.path.exists(time_logger_path):
        with open(time_logger_path, "r") as f:
            time_dict = json.load(f)
    else:
        time_dict = dict()
    if str(args.num) not in time_dict:
        time_dict[str(args.num)] = dict()
    alg = args.algorithm if args.algorithm != "dqn" else ""
    if f"ERMiner{alg}-train" not in time_dict[str(args.num)]:
        time_dict[str(args.num)][f"ERMiner{alg}-train"] = dict()
    time_dict[str(args.num)][f"ERMiner{alg}-train"][str(args.seed)] = end_time-start_time
    with open(time_logger_path, "w") as f:
        json.dump(time_dict, f)
    