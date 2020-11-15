# available NLU models
# from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.policy.multiple_agents import MultipleAgents
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlu.milu.multiwoz import MILU
# available DST models
from convlab2.dst.rule.multiwoz import RuleDST
# from convlab2.dst.mdbt.multiwoz import MDBT
# from convlab2.dst.sumbt.multiwoz import SUMBT
# from convlab2.dst.trade.multiwoz import TRADE
# from convlab2.dst.comer.multiwoz import COMER
# available Policy models
from convlab2.policy.rule.multiwoz import RulePolicy
# from convlab2.policy.ppo.multiwoz import PPOPolicy
# from convlab2.policy.pg.multiwoz import PGPolicy
# from convlab2.policy.mle.multiwoz import MLEPolicy
# from convlab2.policy.vhus.multiwoz import UserPolicyVHUS
# from convlab2.policy.mdrg.multiwoz import MDRGWordPolicy
# from convlab2.policy.hdsa.multiwoz import HDSA
# from convlab2.policy.larl.multiwoz import LaRL
# available NLG models
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.sclstm.multiwoz import SCLSTM
# available E2E models
# from convlab2.e2e.sequicity.multiwoz import Sequicity
# from convlab2.e2e.damd.multiwoz import Damd
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer
from pprint import pprint
import random
import numpy as np
import torch


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():
    # go to README.md of each model for more information
    # BERT nlu
    # sys_nlu = BERTNLU()
    sys_nlu = None
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy1 = MultipleAgents(transfer=True)
    sys_policy2 = MultipleAgents(transfer=False)
    # template NLG
    # sys_nlg = TemplateNLG(is_user=False)
    sys_nlg = None
    # assemble
    sys_agent1 = PipelineAgent(sys_nlu, sys_dst, sys_policy1, sys_nlg, name='sys')
    sys_agent2 = PipelineAgent(sys_nlu, sys_dst, sys_policy2, sys_nlg, name='sys')

    # BERT nlu trained on sys utterance
    # user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
    #                    model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    user_nlu = None
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr', generate_shared=False)
    # template NLG
    # user_nlg = TemplateNLG(is_user=True)
    user_nlg = None
    # assemble
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    analyzer.compare_models(agent_list=[sys_agent1, sys_agent2], model_name=['transfer', 'no-transfer'], total_dialog=500)

if __name__ == '__main__':
    test_end2end()
