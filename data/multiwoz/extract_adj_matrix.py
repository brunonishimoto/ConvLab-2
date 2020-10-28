"""
extract all value appear in dialog act and state, for translation.
"""
import json
import zipfile
import copy
import numpy as np
import pandas as pd
from convlab2.util.multiwoz.state import default_state


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

def domain_slot():
    state = default_state()['belief_state']

    mapping = {}
    inverse_mapping = {}
    counter = 0

    for domain in state:
        for slot_type in state[domain]:
            for slot in state[domain][slot_type]:
                if slot != "booked":
                    domain_slot = f"{domain}-{slot}"
                    mapping[domain_slot] = counter
                    inverse_mapping[counter] = domain_slot
                    counter += 1

    return mapping, inverse_mapping, counter

def include_domain(domain):
    is_domain = False
    for slot_type in domain:
        for slot in domain[slot_type]:
            if domain[slot_type][slot]:
                is_domain = True
                break

    return is_domain

def extract_adj_matrix(data, mapping, adj_matrix, pair_counter):

    for id in data:
        if data[id]['log'][-1]['metadata']:
            metadata = data[id]['log'][-1]['metadata']
        else:
            metadata = data[id]['log'][-2]['metadata']

        domains = [domain for domain in metadata if include_domain(metadata[domain]) and domain != 'bus']

        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                for slot_type1 in metadata[domain1]:
                    for slot1 in metadata[domain1][slot_type1]:
                        for slot_type2 in metadata[domain2]:
                            for slot2 in metadata[domain2][slot_type2]:
                                if slot1 != 'booked' and slot2 != 'booked' and f"{domain1}-{slot1}" in mapping and f"{domain2}-{slot2}" in mapping:
                                    if metadata[domain1][slot_type1][slot1] and metadata[domain2][slot_type2][slot2] and \
                                    metadata[domain1][slot_type1][slot1] != 'not mentioned' and metadata[domain2][slot_type2][slot2] != 'not mentioned' and \
                                    metadata[domain1][slot_type1][slot1] != 'dontcare' and metadata[domain2][slot_type2][slot2] != 'dontcare' and \
                                    metadata[domain1][slot_type1][slot1] != 'none' and metadata[domain2][slot_type2][slot2] != 'none':
                                        if metadata[domain1][slot_type1][slot1] == metadata[domain2][slot_type2][slot2]:
                                            adj_matrix[mapping[f"{domain1}-{slot1}"]][mapping[f"{domain2}-{slot2}"]] += 1
                                            adj_matrix[mapping[f"{domain2}-{slot2}"]][mapping[f"{domain1}-{slot1}"]] += 1
                                        pair_counter[mapping[f"{domain1}-{slot1}"]][mapping[f"{domain2}-{slot2}"]] += 1
                                        pair_counter[mapping[f"{domain2}-{slot2}"]][mapping[f"{domain1}-{slot1}"]] += 1
    # return np.divide(adj_matrix, pair_counter, np.zeros(adj_matrix.shape), where=(pair_counter != 0))
    # return adj_matrix

if __name__ == '__main__':
    mapping, inverse_mapping, counter = domain_slot()

    adj_matrix = np.zeros((len(mapping), len(mapping)))
    pair_counter = copy.deepcopy(adj_matrix)

    for s in ['train']:
        data = read_zipped_json(s + '.json.zip', s + '.json')
    extract_adj_matrix(data, mapping, adj_matrix, pair_counter)

    adj_matrix = np.divide(adj_matrix, pair_counter, np.zeros(adj_matrix.shape), where=(pair_counter != 0))
    df = pd.DataFrame(data=adj_matrix, index=mapping.keys(), columns=mapping.keys())
    df.to_csv('adj_matrix.csv')

    json.dump(mapping, open('node2idx.json', 'w'), indent=2)
    json.dump(inverse_mapping, open('idx2node.json', 'w'), indent=2)
