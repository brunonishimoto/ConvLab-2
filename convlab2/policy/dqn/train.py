# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import logging
import random
import json
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy, RuleBasedMultiwozBot
from convlab2.policy.dqn import DQN
from convlab2.policy.rlmodule import Memory, Transition, MemoryReplay
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

def save_json_file(path, data, mode="w"):
    """Save a json file."""
    try:
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        json.dump(data, open(path, mode), indent=2)
        print(f'saved data in {path}')
    except Exception as e:
        print(f'Error: Writing model fails: {path}')
        print(e)

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()
    info = {'success': [], 'rewards': [], 'turns': [], 'step_rewards': []}

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        total_reward = 0

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            total_reward += r
            if done:
                info['success'].append(env.usr.policy.policy.goal.task_complete())
                info['rewards'].append(total_reward)
                info['turns'].append(real_traj_len)
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff, info])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0, info0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_, info_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
        info0['success'].extend(info_['success'])
        info0['rewards'].extend(info_['rewards'])
        info0['turns'].extend(info_['turns'])
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    metrics = {
        'success_rate': np.average(info0["success"]),
        'avg_reward': np.average(info0["rewards"]),
        'avg_turns': np.average(info0["turns"])
    }

    logging.info(f'batch success rate: {np.average(info0["success"])}')
    logging.info(f'batch avg reward: {np.average(info0["rewards"])}')
    logging.info(f'batch avg turns: {np.average(info0["turns"])}')

    return buff, metrics


def update(env, policy, batchsz, epoch, process_num, experience_replay):
    # sample data asynchronously
    buff, metrics = sample(env, policy, batchsz, process_num)

    experience_replay.append(buff)

    policy.update(epoch, experience_replay)

    return metrics

def run_warmup(env, policy, batchsz, warmup_epoch, process_num, experience_replay, policy_sys):
    # sample data asynchronously
    buff, _ = sample(env, policy, batchsz, process_num)

    experience_replay.append(buff)

    # policy_sys.update(warmup_epoch, experience_replay)

def evaluate(policy_sys, dst_sys, simulator, domains):
    seed = 20201015
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_sys.eval()

    agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

    evaluator = MultiWozEvaluator(domains=domains)
    sess = BiSession(agent_sys, simulator, None, evaluator)

    task_success = {'All': []}
    rewards = []
    turns = []

    for seed in range(100):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sess.init_session()
        sys_response = []
        total_reward = 0
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            total_reward += reward
            if session_over is True:
                task_succ = sess.evaluator.task_success()
                rewards.append(total_reward)
                turns.append(i)
                break
        else:
            rewards.append(total_reward)
            turns.append(40)
            task_succ = 0

        for key in sess.evaluator.goal:
            if key not in task_success:
                task_success[key] = []
            task_success[key].append(task_succ)
        task_success['All'].append(task_succ)

    for key in task_success:
        logging.info(f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')

    metrics = {
        'success_rate': np.average(task_success['All']),
        'avg_reward': np.average(rewards),
        'avg_turns': np.average(turns)
    }

    policy_sys.train()

    return metrics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=100, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=150, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=4, help="number of processes of trajactory sampling")
    parser.add_argument("--warm_up", type=int, default=10, help="number of warm_up episodes (0 if no warm_up)")
    parser.add_argument("--mem_size", type=int, default=10000, help="size of experience replay")
    parser.add_argument("--num_exp", type=int, default=10, help="number repeats for each experiences")
    parser.add_argument("--domain", type=str, default="all", help="domain to train")
    args = parser.parse_args()


    for exp in range(args.num_exp):
        experience_replay = MemoryReplay(args.mem_size)

        # simple rule DST
        dst_sys = RuleDST()

        if args.domain == 'all':
            domains = None
        else:
            domains = [args.domain]

        policy_sys = DQN(True, domains=domains, exp=exp)
        policy_sys.load(args.load_path)

        # rule policy for warm up
        policy_sys_rule = RulePolicy(character='sys', domains=domains)

        policy_sys_rule.vector = policy_sys.vector

        # not use dst
        dst_usr = None
        # rule policy
        policy_usr = RulePolicy(character='usr', domains=domains)
        # assemble
        simulator = PipelineAgent(None, None, policy_usr, None, 'user')

        # evaluator = MultiWozEvaluator(domains=domains)
        evaluator = None
        env = Environment(None, simulator, None, dst_sys, evaluator)

        for i in range(args.warm_up):
            run_warmup(env, policy_sys_rule, args.batchsz, i, args.process_num, experience_replay, policy_sys)
            # if (i+1) % 5 == 0:
            #     logging.info(f"Evaluating warmup: epoch {i}")
            #     evaluate(policy_sys, dst_sys, simulator, domains)

        logging.info('Warmup Ended')

        performance_metrics = {
            'success_rate': {},
            'avg_reward': {},
            'avg_turns': {}
        }

        performance_metrics_test = {
            'success_rate': {},
            'avg_reward': {},
            'avg_turns': {}
        }

        for i in range(args.epoch):
            metrics = update(env, policy_sys, args.batchsz, i, args.process_num, experience_replay)

            performance_metrics['success_rate'][i] = metrics['success_rate']
            performance_metrics['avg_reward'][i] = metrics['avg_reward']
            performance_metrics['avg_turns'][i] = metrics['avg_turns']

            logging.info(f"Evaluating: epoch {i}")
            metrics_test = evaluate(policy_sys, dst_sys, simulator, domains)

            performance_metrics_test['success_rate'][i] = metrics_test['success_rate']
            performance_metrics_test['avg_reward'][i] = metrics_test['avg_reward']
            performance_metrics_test['avg_turns'][i] = metrics_test['avg_turns']

            save_json_file(f'./checkpoints/{args.domain}/{exp}/train.json', performance_metrics)
            save_json_file(f'./checkpoints/{args.domain}/{exp}/evaluate.json', performance_metrics_test)
