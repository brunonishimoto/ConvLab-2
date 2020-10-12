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
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

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

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0
    info = {'success': [], 'rewards': [], 'turns': []}
    success = []
    rewards = []

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
            real_traj_len = t + 1

            total_reward += r
            if done:
                info['success'].append(env.usr.policy.policy.goal.task_complete())
                info['rewards'].append(total_reward)
                info['turns'].append(2 * real_traj_len)
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

    logging.info(f'batch success rate: {sum(info0["success"]) / len(info0["success"])}')
    logging.info(f'batch avg reward: {sum(info0["rewards"]) / len(info0["rewards"])}')
    logging.info(f'batch avg turns: {sum(info0["turns"]) / len(info0["turns"])}')

    return buff.get_batch()

def evaluate(policy_sys, dst_sys, simulator, domains):
    seed = 20190827
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_sys.eval()

    agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

    evaluator = MultiWozEvaluator(domains=domains)
    sess = BiSession(agent_sys, simulator, None, evaluator)

    task_success = {'All': []}
    for seed in range(100):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sess.init_session()
        sys_response = []
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            if session_over is True:
                task_succ = sess.evaluator.task_success()
                break
        else:
            task_succ = 0

        for key in sess.evaluator.goal:
            if key not in task_success:
                task_success[key] = []
            task_success[key].append(task_succ)
        task_success['All'].append(task_succ)

    for key in task_success:
        logging.info(f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')

    policy_sys.train()

def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, r, mask)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=4, help="number of processes of trajactory sampling")
    args = parser.parse_args()

    # simple rule DST
    dst_sys = RuleDST()

    domains = ['taxi']
    # domains = None

    policy_sys = PPO(True, domains=None)
    policy_sys.load(args.load_path)

    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = RulePolicy(character='usr', domains=domains)
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    # evaluator = MultiWozEvaluator(domains=domains)
    evaluator = None
    env = Environment(None, simulator, None, dst_sys, evaluator)

    for i in range(args.epoch):
        update(env, policy_sys, args.batchsz, i, args.process_num)
        if (i+1) % 5 == 0:
            logging.info(f"Evaluating: epoch {i}")
            evaluate(policy_sys, dst_sys, simulator, domains)
