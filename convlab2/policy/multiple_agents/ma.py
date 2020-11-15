# -*- coding: utf-8 -*-
import torch
from torch import optim
from torch import nn
import numpy as np
import numpy.ma as ma
import logging
import os
import json
import copy
from convlab2.policy.policy import Policy
from convlab2.policy.dqn import DQN
from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import EpsilonGreedyPolicy, DiscretePolicy
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
from data.multiwoz.graph_embedding import GraphEmbedding
import zipfile
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS = {
    'restaurant': 0,
    'hotel': 1,
    'taxi': 2,
    'police': 3,
    'train': 4,
    'attraction': 5,
    'hospital': 6
}

SHARED_SLOTS = {
    frozenset(['restaurant', 'hotel']): {'semi': ['pricerange', 'area'], 'book': ['people', 'day']},
    frozenset(['restaurant', 'attraction']): {'semi': ['area']},
    frozenset(['hotel', 'attraction']): {'semi': ['area']},
    frozenset(['restaurant', 'train']): {'book': ['people']},
    frozenset(['hotel', 'train']): {'book': ['people']},
}

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_DNC1 = "do n't care"  # Do not care
DEF_VAL_NUL = 'none'  # for none
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_DNC1]

class MultipleAgents(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz', transfer=None):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)

        self.is_train = is_train
        self.transfer = transfer if transfer != None else cfg['transfer']
        self.cur_domains = None
        self.old_domains = []

        if cfg['use_embedding']:
            self.graph_embedding = GraphEmbedding(50)
            self.graph_embedding.load_embedding(os.path.join(root_dir, 'data/multiwoz/embeddings_all.mdl'))
            self.embeddings = self.graph_embedding.load_embedding

        self.agents = []
        for domain in DOMAINS:
            if cfg['policy'] == "DQN":
                self.agents.append(DQN(is_train=False, domains=None))
                self.agents[DOMAINS[domain]].load(f'../dqn/save_{domain}/99')
            elif cfg['policy'] == "PPO":
                self.agents.append(PPO(is_train=False, domains=[domain]))
                self.agents[DOMAINS[domain]].load(f'../ppo/save/{domain}/1/best_ppo')
            else:
                raise Exception("Error: the given model is not supported")

    def predict(self, state):
        domains = self.extract_domains(state)

        if domains:
            if self.cur_domains and self.cur_domains not in self.old_domains:
                self.old_domains.append(self.cur_domains)
            self.cur_domains = domains

        action = []

        usr_action = state['user_action']
        gave_name = False
        for act in usr_action:
            if act[0].lower() == 'inform' and act[2].lower() == 'name':
                gave_name = True


        if self.transfer and self.old_domains and self.cur_domains not in self.old_domains and not gave_name:
            action.extend(self.transfer_inter_domain(state))
            for act in action:
                if act[2] == 'Pricerange':
                    act[2] = 'Price'

        if True:
            cur_agents = self.agents[DOMAINS[self.cur_domains]]

            action.extend(self.select_domain_action(cur_agents, state))

        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        self.cur_domains = None
        self.old_domains = []

    def extract_domains(self, state):
        usr_action = state['user_action']

        for act in usr_action:
            act_domain = act[1].lower()
            if act_domain in DOMAINS:
                return act_domain

        return None

    def select_domain_action(self, agents, state):
        action = agents.predict(copy.deepcopy(state))

        # state['system_action'] = action
        return action

    def transfer_common_slots(self, old, new, state):
        action = []
        for slot in SHARED_SLOTS[frozenset([old, new])].get('semi', []):
            if not state['belief_state'][new]['semi'][slot] and state['belief_state'][old]['semi'][slot] not in NOT_SURE_VALS:
                state['belief_state'][new]['semi'][slot] = state['belief_state'][old]['semi'][slot]
                if state['belief_state'][old]['semi'][slot]:
                    action.append(['Confirm', new.capitalize(), slot.capitalize(), state['belief_state'][old]['semi'][slot]])

        for slot in SHARED_SLOTS[frozenset([old, new])].get('book', []):
            if not state['belief_state'][new]['book'][slot] and state['belief_state'][old]['book'][slot] not in NOT_SURE_VALS:
                state['belief_state'][new]['book'][slot] = state['belief_state'][old]['book'][slot]
                if state['belief_state'][old]['book'][slot]:
                    action.append(['Confirm', new.capitalize(), slot.capitalize(), state['belief_state'][old]['book'][slot]])

        return action

    def transfer_taxi_restaurant(self, other_domain, state):
        action = []
        if not state['belief_state']['taxi']['semi']['arriveBy'] and state['belief_state']['restaurant']['book']['time'] not in NOT_SURE_VALS:
            state['belief_state']['taxi']['semi']['arriveBy'] = state['belief_state']['restaurant']['book']['time']
            if state['belief_state']['restaurant']['book']['time']:
                action.append(['Confirm', 'Taxi', 'Arrive', state['belief_state']['restaurant']['book']['time']])

        if not state['belief_state']['taxi']['semi']['destination'] and state['belief_state']['restaurant']['semi']['name'] not in NOT_SURE_VALS:
            state['belief_state']['taxi']['semi']['destination'] = state['belief_state']['restaurant']['semi']['name']
            if state['belief_state']['restaurant']['semi']['name']:
                action.append(['Confirm', 'Taxi', 'Dest', state['belief_state']['restaurant']['semi']['name']])

        if not state['belief_state']['taxi']['semi']['departure'] and state['belief_state'][other_domain]['semi']['name'] not in NOT_SURE_VALS:
            state['belief_state']['taxi']['semi']['departure'] = state['belief_state'][other_domain]['semi']['name']
            if state['belief_state'][other_domain]['semi']['name']:
                action.append(['Confirm', 'Taxi', 'Depart', state['belief_state'][other_domain]['semi']['name']])

        return action

    def transfer_inter_domain(self, state):
        action = []
        if 'restaurant' in self.old_domains and 'hotel' == self.cur_domains:
            action.extend(self.transfer_common_slots('restaurant', 'hotel', state))

        elif 'hotel' in self.old_domains and 'restaurant' == self.cur_domains:
            action.extend(self.transfer_common_slots('hotel', 'restaurant', state))

        if 'restaurant' in self.old_domains and 'attraction' == self.cur_domains:
            action.extend(self.transfer_common_slots('restaurant', 'attraction', state))

        elif 'attraction' in self.old_domains and 'restaurant' == self.cur_domains:
            action.extend(self.transfer_common_slots('attraction', 'restaurant', state))

        if 'attraction' in self.old_domains and 'hotel' == self.cur_domains:
            action.extend(self.transfer_common_slots('attraction', 'hotel', state))

        elif 'hotel' in self.old_domains and 'attraction' == self.cur_domains:
            action.extend(self.transfer_common_slots('hotel', 'attraction', state))

        if 'restaurant' in self.old_domains and 'train' == self.cur_domains:
            action.extend(self.transfer_common_slots('restaurant', 'train', state))
            if not state['belief_state']['train']['semi']['day'] and state['belief_state']['restaurant']['book']['day'] not in NOT_SURE_VALS:
                state['belief_state']['train']['semi']['day'] = state['belief_state']['restaurant']['book']['day']
                if state['belief_state']['restaurant']['book']['day']:
                    action.append(['Confirm', 'Train', 'Day', state['belief_state']['restaurant']['book']['day']])

        elif 'train' in self.old_domains and 'restaurant' == self.cur_domains:
            action.extend(self.transfer_common_slots('train', 'restaurant', state))
            if not state['belief_state']['restaurant']['book']['day'] and state['belief_state']['restaurant']['book']['day'] not in NOT_SURE_VALS:
                state['belief_state']['restaurant']['book']['day'] = state['belief_state']['train']['semi']['day']
                if state['belief_state']['train']['semi']['day']:
                    action.append(['Confirm', 'Restaurant', 'Day', state['belief_state']['train']['semi']['day']])

        if 'hotel' in self.old_domains and 'train' == self.cur_domains:
            action.extend(self.transfer_common_slots('hotel', 'train', state))
            if not state['belief_state']['train']['semi']['day'] and state['belief_state']['hotel']['book']['day'] not in NOT_SURE_VALS:
                state['belief_state']['train']['semi']['day'] = state['belief_state']['hotel']['book']['day']
                if state['belief_state']['hotel']['book']['day']:
                    action.append(['Confirm', 'Train', 'Day', state['belief_state']['hotel']['book']['day']])

        elif 'train' in self.old_domains and 'hotel' == self.cur_domains:
            action.extend(self.transfer_common_slots('train', 'hotel', state))
            if not state['belief_state']['hotel']['book']['day'] and state['belief_state']['train']['semi']['day'] not in NOT_SURE_VALS:
                state['belief_state']['hotel']['book']['day'] = state['belief_state']['train']['semi']['day']
                if state['belief_state']['train']['semi']['day']:
                    action.append(['Confirm', 'Hotel', 'Day', state['belief_state']['train']['semi']['day']])

        # elif 'taxi' in self.cur_domains:
        #     if len(self.old_domains) == 2:
        #         dom1, dom2 = self.old_domains
        #         if dom1 == 'restaurant':
        #             action = self.transfer_taxi_restaurant(dom2, state)
        #         elif dom2 == 'restaurant':
        #             action = self.transfer_taxi_restaurant(dom1, state)
        #         else:
        #             action = []
        #             if not state['belief_state']['taxi']['semi']['destination'] and state['belief_state'][dom2]['semi']['name'] not in NOT_SURE_VALS:
        #                 state['belief_state']['taxi']['semi']['destination'] = state['belief_state'][dom2]['semi']['name']
        #                 if state['belief_state'][dom2]['semi']['name']:
        #                     action.append(['Confirm', 'Taxi', 'Dest', state['belief_state'][dom2]['semi']['name']])

        #             if not state['belief_state']['taxi']['semi']['departure'] and state['belief_state'][dom1]['semi']['name'] not in NOT_SURE_VALS:
        #                 state['belief_state']['taxi']['semi']['departure'] = state['belief_state'][dom1]['semi']['name']
        #                 if state['belief_state'][dom1]['semi']['name']:
        #                     action.append(['Confirm', 'Taxi', 'Depart', state['belief_state'][dom1]['semi']['name']])

        return action


    def load(self, filename):
        # TODO: load agents from individual agents
        pass
        # dqn_mdl_candidates = [
        #     filename + '_dqn.pol.mdl',
        #     filename + '.pol.mdl',
        #     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dqn', filename + '_dqn.pol.mdl'),
        #     os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
        # ]
        # for dqn_mdl in dqn_mdl_candidates:
        #     if os.path.exists(dqn_mdl):
        #         self.net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
        #         self.target_net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
        #         logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(dqn_mdl))
        #         break
