# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database

mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'price': 'pricerange'},
           'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name',
                     'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
           'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'type': 'type'},
           'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination',
                     'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
           'taxi': {'car': 'car type', 'phone': 'phone'},
           'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
           'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}

# Information required to finish booking, according to different domain.
booking_info = {'Train': ['People'],
                'Restaurant': ['Time', 'Day', 'People'],
                'Hotel': ['Stay', 'Day', 'People']}

DEFAULT_INTENT_FILEPATH = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                            'data/multiwoz/trackable_intent.json'
                        )

class MultiWozVector(Vector):

    def __init__(self, voc_file, voc_opp_file, character='sys',
                 intent_file=DEFAULT_INTENT_FILEPATH,
                 domains=None,
                 composite_actions=False,
                 vocab_size=500):

        self.domains = domains
        self.belief_domains = [d.capitalize() for d in domains] if domains else['Attraction', 'Restaurant', 'Train', 'Hotel', 'Taxi', 'Hospital', 'Police']
        self.db_domains = list(set(['Attraction', 'Restaurant', 'Train', 'Hotel']) & set(self.belief_domains))

        self.composite_actions = composite_actions
        self.vocab_size = vocab_size

        with open(intent_file) as f:
            intents = json.load(f)
        self.informable = intents['informable']
        self.requestable = intents['requestable']
        self.db = Database(domains=domains)

        self.all_booking_slots = ['Ref', 'none', 'Name']
        for domain in self.belief_domains:
            self.all_booking_slots.extend(booking_info.get(domain, []))

        with open(voc_file) as f:
            self.da_voc = f.read().splitlines()
        with open(voc_opp_file) as f:
            self.da_voc_opp = f.read().splitlines()

        self.da_voc = self.filter_da_by_domains(self.da_voc)
        self.da_voc_opp = self.filter_da_by_domains(self.da_voc_opp)

        if self.composite_actions:
            self.da_voc = [da for da in self.da_voc if not da.endswith(('2', '3', '4', '5'))]
            self.load_composite_actions()


        self.character = character
        self.generate_dict()
        self.cur_domain = None

    def filter_da_by_domains(self, da_voc):
        filtered_da_voc = []
        for da in da_voc:
            in_domain = self.is_in_domains(da)
            if in_domain:
                filtered_da_voc.append(da)
        return filtered_da_voc

    def is_in_domains(self, da):
        in_domain = True

        actions = da.split(';')
        for action in actions:
            domain, intent, slot, value = action.split('-')
            if domain.capitalize() not in self.belief_domains and domain != 'general' and not (domain.capitalize() == 'Booking' and slot in self.all_booking_slots):
                in_domain = False
                break

        return in_domain

    def load_composite_actions(self):
        """
        load the composite actions to self.da_voc
        """
        composite_actions_filepath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    'data/multiwoz/da_slot_cnt.json')
        with open(composite_actions_filepath, 'r') as f:
            composite_actions_stats = json.load(f)
            for action in composite_actions_stats:
                if len(action.split(';')) > 1 and self.is_in_domains(action):
                    # append only composite actions as single actions are already in self.da_voc
                    self.da_voc.append(action)

                if len(self.da_voc) == self.vocab_size:
                    break

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)
        self.opp2vec = dict((a, i) for i, a in enumerate(self.da_voc_opp))
        self.da_opp_dim = len(self.da_voc_opp)

        self.belief_state_dim = 0
        for domain in self.belief_domains:
            for slot, value in default_state(self.domains)['belief_state'][domain.lower()]['semi'].items():
                self.belief_state_dim += 1

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
                         len(self.db_domains) + 6 * len(self.db_domains) + 1

    def pointer(self, turn):
        pointer_vector = np.zeros(6 * len(self.db_domains))
        for domain in self.db_domains:
            constraint = turn[domain.lower()]['semi'].items()
            entities = self.db.query(domain.lower(), constraint)
            pointer_vector = self.one_hot_vector(len(entities), domain, pointer_vector)

        return pointer_vector

    def one_hot_vector(self, num, domain, vector):
        """Return number of available entities for particular domain."""
        if domain != 'train':
            idx = self.db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        else:
            idx = self.db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num <= 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num <= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num <= 10:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num <= 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num > 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

        return vector

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (dict):
                Dialog state
            action (tuple):
                Dialog act
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        self.state = state['belief_state']

        # when character is sys, to help query database when da is booking-book
        # update current domain according to user action
        if self.character == 'sys':
            action = state['user_action']
            for intent, domain, slot, value in action:
                if domain in self.db_domains:
                    self.cur_domain = domain

        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                opp_act_vec[self.opp2vec[da]] = 1.

        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        last_act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                last_act_vec[self.act2vec[da]] = 1.

        belief_state = np.zeros(self.belief_state_dim)
        i = 0
        for domain in self.belief_domains:
            for slot, value in state['belief_state'][domain.lower()]['semi'].items():
                if value:
                    belief_state[i] = 1.
                i += 1

        book = np.zeros(len(self.db_domains))
        for i, domain in enumerate(self.db_domains):
            if state['belief_state'][domain.lower()]['book']['booked']:
                book[i] = 1.

        degree = self.pointer(state['belief_state'])

        final = 1. if state['terminated'] else 0.

        state_vec = np.r_[opp_act_vec, last_act_vec, belief_state, book, degree, final]
        assert len(state_vec) == self.state_dim
        return state_vec


    def dbquery_domain(self, domain):
        """
        query entities of specified domain
        Args:
            domain string:
                domain to query
        Returns:
            entities list:
                list of entities of the specified domain
        """
        constraint = self.state[domain.lower()]['semi'].items()
        return self.db.query(domain.lower(), constraint)

    def action_devectorize(self, action_vec):
        """
        recover an action
        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple):
                Dialog act
        """
        act_array = []

        if self.composite_actions:
            act_idx = np.argmax(action_vec)
            act_array = self.vec2act[act_idx].split(';')
        else:
            for i, idx in enumerate(action_vec):
                if idx == 1:
                    act_array.append(self.vec2act[i])
        action = deflat_da(act_array)
        entities = {}
        for domint in action:
            domain, intent = domint.split('-')
            if domain not in entities and domain.lower() not in ['general', 'booking']:
                entities[domain] = self.dbquery_domain(domain)
        if self.cur_domain and self.cur_domain not in entities:
            entities[self.cur_domain] = self.dbquery_domain(self.cur_domain)
        action = lexicalize_da(action, entities, self.state, self.requestable, self.cur_domain)
        return action

    def action_vectorize(self, action):
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        act_vec = np.zeros(self.da_dim)

        if action:
            if self.composite_actions:
                for act in self.act2vec:
                    if action == act.split(';'):
                        act_vec[self.act2vec[act]] = 1.
                        break
                else:
                    most_similar_action = self.find_closest_action(action)
                    if most_similar_action:
                        act_vec[self.act2vec[most_similar_action]] = 1.
            else:
                for da in action:
                    if da in self.act2vec:
                        act_vec[self.act2vec[da]] = 1.
        return act_vec

    def find_closest_action(self, action):
        candidates = []
        best_intersection = 0
        smaller_length = 0

        for da in self.da_voc:
            da_splited = da.split(';')
            intersection = len(set(action) & set(da_splited))
            if intersection > best_intersection:
                best_intersection = intersection
                candidates = [da]
                smaller_length = len(da_splited)
            elif intersection == best_intersection and intersection > 0:
                length = len(da_splited)
                if length < smaller_length:
                    smaller_length = length
                    candidates = [da]
                elif length == smaller_length:
                    candidates.append(da)

        return np.random.choice(candidates) if candidates else None
