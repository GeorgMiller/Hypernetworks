import numpy as np
from doudizhu.utils import opt_legal


class DummyRuleAgent(object):
    """
    An agent using a dummy rule: if a solo card could be played in any of the comb: 'bomb', 'trio', 'trio_chain',
    'solo_chain', 'pair_chain', 'pair', remove the 'solo' action from legal_actions. Then choose the minimum action_id
    from legal_actions.

    ### ~25% winning rate for landlord against DouDizhuRuleAgentV1
    """

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''

        opt_legal_actions = opt_legal(state['legal_actions'])
        return np.min(opt_legal_actions)
        #return np.random.choice(opt_legal_actions)

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])
        return self.step(state), probs
