# -*- coding: utf-8 -*-
import torch
from torch import optim
from torch import nn
import numpy as np
import logging
import os
import json
import copy
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import EpsilonGreedyPolicy, MemoryReplay, SoftmaxPolicy
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz', domains=None):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_tau.json'), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.save_per_epoch = cfg['save_per_epoch']
        self.training_iter = cfg['training_iter']
        self.training_batch_iter = cfg['training_batch_iter']
        self.batch_size = cfg['batch_size']
        self.gamma = cfg['gamma']
        self.update_frequency = cfg['update_frequency']
        self.is_train = is_train
        if is_train:
            init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))

        # construct multiwoz vector
        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file, composite_actions=True, vocab_size=cfg['vocab_size'], domains=domains)

        self.net = SoftmaxPolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim, cfg['tau_spec']).to(device=DEVICE)
        self.target_net = copy.deepcopy(self.net)

        self.online_net = self.net
        self.eval_net = self.target_net

        if is_train:
            self.net_optim = optim.Adam(self.net.parameters(), lr=cfg['lr'])
            self.loss_fn = nn.MSELoss()

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.net.select_action(s_vec.to(device=DEVICE), self.is_train)

        action = self.vector.action_devectorize(a.numpy())

        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
        next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)

        q_preds = self.net(s)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_s)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_s)

        act_q_preds = q_preds.gather(-1, a.argmax(-1).long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = r + self.gamma * mask * max_next_q_preds


        q_loss = self.loss_fn(act_q_preds, max_q_targets)

        return q_loss

    def update(self, epoch, experience_replay):
        total_loss = 0.
        # self.training_iter = len(experience_replay) // self.batch_size
        for _ in range(self.training_iter):
            # 1. batch a sample from memory
            batch = experience_replay.get_batch(batch_size=self.batch_size)

            for _ in range(self.training_batch_iter):
                # 2. calculate the Q loss
                loss = self.calc_q_loss(batch)

                # 3. make a optimization step
                self.net_optim.zero_grad()
                loss.backward()
                self.net_optim.step()

                total_loss += loss.item()

            # logging.debug('<<dialog policy dqn>> epoch {}, iteration {}, loss {}'.format(epoch, i, round_loss / self.training_batch_iter))
        total_loss /= (self.training_batch_iter * self.training_iter)
        logging.debug('<<dialog policy dqn>> epoch {}, total_loss {}'.format(epoch, total_loss))

        # update the epsilon value
        if self.is_train:
            self.net.update(epoch)

        # update the target network
        if epoch % self.update_frequency == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)

    def eval(self):
        self.is_train = False

    def train(self):
        self.is_train = True

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.net.state_dict(), directory + '/' + str(epoch) + '_dqn.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        dqn_mdl_candidates = [
            filename + '_dqn.pol.mdl',
            filename + '.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_dqn.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
        ]
        for dqn_mdl in dqn_mdl_candidates:
            if os.path.exists(dqn_mdl):
                self.net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                self.target_net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(dqn_mdl))
                break
