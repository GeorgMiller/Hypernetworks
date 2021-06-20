import numpy as np
from doudizhu.utils import get_history, get_hand_length
from envs.env import Env


class DoudizhuEnv(Env):
    '''
    Doudizhu Environment
    '''

    def __init__(self, config, state_shape, type=None):
        from doudizhu.utils import SPECIFIC_MAP, CARD_RANK_STR
        from doudizhu.utils import ACTION_LIST, ACTION_SPACE
        from doudizhu.utils import encode_cards, encode_cards_conv
        from doudizhu.utils import cards2str, cards2str_with_suit
        from doudizhu.game import DoudizhuGame as Game
        if config['use_conv']:
            self._encode_cards = encode_cards_conv
        else:
            self._encode_cards = encode_cards
        self._cards2str = cards2str
        self._cards2str_with_suit = cards2str_with_suit
        self._SPECIFIC_MAP = SPECIFIC_MAP
        self._CARD_RANK_STR = CARD_RANK_STR
        self._ACTION_LIST = ACTION_LIST
        self._ACTION_SPACE = ACTION_SPACE

        self.name = 'doudizhu'
        self.game = Game()
        super().__init__(config)
        self.state_shape = state_shape
        self.type = type

    def _extract_state(self, state):
        ###### only changed this function and added get_hand_length func in mydoudizhu.py, the rest is the same ######
        ''' Encode state
        Args:
            state (dict): dict of original state
        Returns:
            numpy array:  n*5*15 array:
                            type: None
                                n=5: cur_hand, others_hand, last_2_actions, played_cards
                                n=7: cur_hand, others_hand, last_2_actions, hist_3

                            type: 'cooperation'
                                n=7: cur_hand, others_hand, last_2_actions, role, hand_length, played_cards
                                n=9: cur_hand, others_hand, last_2_actions, hist_3, role, hand_length

                            current hand
                            the union of the other two players' hand
                            the recent two actions
                            a history(played cards) of current, next, over next player
                            played cards of all players
                            player's role: 0s if landlord, -1s for up peasant, 1s for down peasant
                            current hand_length of current, next, over next player

        '''
        obs = np.zeros(tuple(self.state_shape), dtype=int)

        current_hand = self.state_cur_hand(state)
        others_hand = self.state_others_hand(state)
        last_action = self.state_last_actions(state)
        played_cards = self.state_played_cards(state)
        hist = self.state_hist(state)
        role = self.state_role(state)
        hand_length = self.state_hand_length(state)

        if self.type == 'cooperation':
            # simple cooperation
            if self.state_shape[0] == 7:
                obs = np.concatenate((current_hand, others_hand, last_action, played_cards, role, hand_length), 0)

                # complicated cooperation
            elif self.state_shape[0] == 9:
                obs = np.concatenate((current_hand, others_hand, last_action, hist, role, hand_length), 0)
        else:
            if self.state_shape[0] == 5:
                # simple
                obs = np.concatenate((current_hand, others_hand, last_action, played_cards), 0)
                # complicated
            elif self.state_shape[0] == 7:
                obs = np.concatenate((current_hand, others_hand, last_action, hist), 0)

        extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions()}
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            # TODO: state['actions'] can be None, may have bugs
            if state['actions'] == None:
                extracted_state['raw_legal_actions'] = []
            else:
                extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def state_cur_hand(self, state):
        current_hand = np.zeros(self.state_shape[1:], int)
        self._encode_cards(current_hand, state['current_hand'])
        return np.expand_dims(current_hand, 0)

    def state_others_hand(self, state):
        others_hand = np.zeros(self.state_shape[1:], int)
        self._encode_cards(others_hand, state['others_hand'])
        return np.expand_dims(others_hand, 0)

    def state_last_actions(self, state):
        last_action = np.zeros((2, self.state_shape[1], self.state_shape[2]), int)
        for i, action in enumerate(state['trace'][-2:]):
            if action[1] != 'pass':
                self._encode_cards(last_action[1 - i], action[1])
        return last_action

    def state_played_cards(self, state):
        played_cards = np.zeros(self.state_shape[1:], int)
        if state['played_cards'] is not None:
            self._encode_cards(played_cards, state['played_cards'])
        return np.expand_dims(played_cards, 0)

    def state_hist(self, state):
        hist = np.zeros((3, self.state_shape[1], self.state_shape[2]), int)
        history = get_history(state)
        for i in range(self.player_num):
            self._encode_cards(hist[i], history[(state['self'] + i + 3) % 3])
        return hist

    def state_role(self, state):
        role = np.zeros(self.state_shape[1:], int)
        if state['self'] == 0:
            role[:] = np.zeros(15, dtype=int)
        elif state['self'] == 1:
            role[:] = np.ones(15, dtype=int)
        elif state['self'] == 2:
            role[:] = np.ones(15, dtype=int) * -1
        return np.expand_dims(role, 0)

    def state_hand_length(self, state):

        left_cards = get_hand_length(state)
        hand_length = np.zeros(self.state_shape[1:], int)
        for i in range(self.player_num):
            for j in range(left_cards[(state['self'] + i) % 3]):
                hand_length[i][j] = 1
        return np.expand_dims(hand_length, 0)

    def get_payoffs(self):
        '''
        Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def get_winner_id(self):
        return self.game.get_winner_id()

    def _decode_action(self, action_id):
        '''
        Action id -> the action in the doudizhu. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the doudizhu engine.
        '''
        abstract_action = self._ACTION_LIST[action_id]
        # without kicker
        if '*' not in abstract_action:
            return abstract_action
        # with kicker
        legal_actions = self.game.state['actions']
        specific_actions = []
        kickers = []
        for legal_action in legal_actions:
            for abstract in self._SPECIFIC_MAP[legal_action]:
                main = abstract.strip('*')
                if abstract == abstract_action:
                    specific_actions.append(legal_action)
                    kickers.append(legal_action.replace(main, '', 1))
                    break
        # choose kicker with minimum score
        player_id = self.game.get_player_id()
        kicker_scores = []
        for kicker in kickers:
            score = 0
            for action in self.game.judger.playable_cards[player_id]:
                if kicker in action:
                    score += 1
            kicker_scores.append(score + self._CARD_RANK_STR.index(kicker[0]))
        min_index = 0
        min_score = kicker_scores[0]
        for index, score in enumerate(kicker_scores):
            if score < min_score:
                min_score = score
                min_index = index
        return specific_actions[min_index]

    def _get_legal_actions(self):
        '''
        Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_action_id = []
        legal_actions = self.game.state['actions']
        if legal_actions:
            for action in legal_actions:
                for abstract in self._SPECIFIC_MAP[action]:
                    action_id = self._ACTION_SPACE[abstract]
                    if action_id not in legal_action_id:
                        legal_action_id.append(action_id)
        return legal_action_id

    def get_perfect_information(self):
        '''
        Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['hand_cards_with_suit'] = [self._cards2str_with_suit(player.current_hand) for player in self.game.players]
        state['hand_cards'] = [self._cards2str(player.current_hand) for player in self.game.players]
        state['landlord'] = self.game.state['landlord']
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state
