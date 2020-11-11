import os
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.mle.loader import ActMLEPolicyDataLoader

class ActMLEPolicyDataLoaderMultiWoz(ActMLEPolicyDataLoader):

    def __init__(self, cfg):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file, domains=cfg['domains'], composite_actions=cfg['composite_actions'], vocab_size=cfg['vocab_size'])

        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['data_dir'])
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir, cfg['domains'])
