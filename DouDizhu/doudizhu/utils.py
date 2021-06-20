'''
Doudizhu utils
'''

import os
import json
from collections import OrderedDict
import threading
import collections
import numpy as np

# Read required docs
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# a map of action to abstract action
with open(os.path.join(ROOT_PATH, 'jsondata/specific_map.json'), 'r') as file:
    SPECIFIC_MAP = json.load(file, object_pairs_hook=OrderedDict)

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

# a map of card to its type. Also return both dict and list to accelerate
with open(os.path.join(ROOT_PATH, 'jsondata/card_type.json'), 'r') as file:
    data = json.load(file, object_pairs_hook=OrderedDict)
    CARD_TYPE = (data, list(data), set(data))

# a map of type to its cards
with open(os.path.join(ROOT_PATH, 'jsondata/type_card.json'), 'r') as file:
    TYPE_CARD = json.load(file, object_pairs_hook=OrderedDict)

# rank list of solo character of cards
CARD_RANK_STR = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
                 'A', '2', 'B', 'R']
CARD_RANK_STR_INDEX = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                       '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                       'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}

INDEX_CARD_RANK_STR = {0: '3', 1: '4', 2: '5', 3: '6', 4: '7',
                       5: '8', 6: '9', 7: 'T', 8: 'J', 9: 'Q',
                       10: 'K', 11: 'A', 12: '2', 13: 'B', 14: 'R'}
# rank list
CARD_RANK = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
             'A', '2', 'BJ', 'RJ']

INDEX = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
         '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
         'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}
INDEX = OrderedDict(sorted(INDEX.items(), key=lambda t: t[1]))

ACTION_ID_TO_STR = {index: action for action, index in ACTION_SPACE.items()}


def doudizhu_sort_str(card_1, card_2):
    '''
    Compare the rank of two cards of str representation

    Args:
        card_1 (str): str representation of solo card
        card_2 (str): str representation of solo card

    Returns:
        int: 1(card_1 > card_2) / 0(card_1 = card2) / -1(card_1 < card_2)
    '''
    key_1 = CARD_RANK_STR.index(card_1)
    key_2 = CARD_RANK_STR.index(card_2)
    if key_1 > key_2:
        return 1
    if key_1 < key_2:
        return -1
    return 0


def doudizhu_sort_card(card_1, card_2):
    '''
    Compare the rank of two cards of Card object

    Args:
        card_1 (object): object of Card
        card_2 (object): object of Card
    '''
    key = []
    for card in [card_1, card_2]:
        if card.rank == '':
            key.append(CARD_RANK.index(card.suit))
        else:
            key.append(CARD_RANK.index(card.rank))
    if key[0] > key[1]:
        return 1
    if key[0] < key[1]:
        return -1
    return 0


def get_landlord_score(current_hand):
    '''
    Roughly judge the quality of the hand, and provide a score as basis to
    bid landlord.

    Args:
        current_hand (str): string of cards. Eg: '56888TTQKKKAA222R'

    Returns:
        int: score
    '''
    score_map = {'A': 1, '2': 2, 'B': 3, 'R': 4}
    score = 0
    # rocket
    if current_hand[-2:] == 'BR':
        score += 8
        current_hand = current_hand[:-2]
    length = len(current_hand)
    i = 0
    while i < length:
        # bomb
        if i <= (length - 4) and current_hand[i] == current_hand[i + 3]:
            score += 6
            i += 4
            continue
        # 2, Black Joker, Red Joker
        if current_hand[i] in score_map:
            score += score_map[current_hand[i]]
        i += 1
    return score


def get_optimal_action(probs, legal_actions, np_random):
    '''
    Determine the optimal action from legal actions
    according to the probabilities of abstract actions.

    Args:
        probs (list): list of probabilities of abstract actions
        legal_actions (list): list of legal actions

    Returns:
        str: optimal legal action
    '''
    abstract_actions = [SPECIFIC_MAP[action] for action in legal_actions]
    action_probs = []
    for actions in abstract_actions:
        max_prob = -1
        for action in actions:
            prob = probs[ACTION_SPACE[action]]
            if prob > max_prob:
                max_prob = prob
        action_probs.append(max_prob)
    optimal_prob = max(action_probs)
    optimal_actions = [legal_actions[index] for index,
                                                prob in enumerate(action_probs) if prob == optimal_prob]
    if len(optimal_actions) > 1:
        return np_random.choice(optimal_actions)
    return optimal_actions[0]


def cards2str_with_suit(cards):
    '''
    Get the corresponding string representation of cards with suit

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    '''
    return ' '.join([card.suit + card.rank for card in cards])


def cards2str(cards):
    '''
    Get the corresponding string representation of cards

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    '''
    response = ''
    for card in cards:
        if card.rank == '':
            response += card.suit[0]
        else:
            response += card.rank
    return response


class LocalObjs(threading.local):
    def __init__(self):
        self.cached_candidate_cards = None


_local_objs = LocalObjs()


def contains_cards(candidate, target):
    '''
    Check if cards of candidate contains cards of target.

    Args:
        candidate (string): A string representing the cards of candidate
        target (string): A string representing the number of cards of target

    Returns:
        boolean
    '''
    # In normal cases, most continuous calls of this function
    #   will test different targets against the same candidate.
    # So the cached counts of each card in candidate can speed up
    #   the comparison for following tests if candidate keeps the same.
    if not _local_objs.cached_candidate_cards or _local_objs.cached_candidate_cards != candidate:
        _local_objs.cached_candidate_cards = candidate
        cards_dict = collections.defaultdict(int)
        for card in candidate:
            cards_dict[card] += 1
        _local_objs.cached_candidate_cards_dict = cards_dict
    cards_dict = _local_objs.cached_candidate_cards_dict

    if target == '':
        return True
    curr_card = target[0]
    curr_count = 1

    for card in target[1:]:
        if card != curr_card:
            if cards_dict[curr_card] < curr_count:
                return False
            curr_card = card
            curr_count = 1
        else:
            curr_count += 1

    if cards_dict[curr_card] < curr_count:
        return False
    return True


# def encode_cards_5x15(plane, cards):
#     '''
#     Encode cards and represent it into plane.
#
#     Args:
#         cards (list or str): list or str of cards, every entry is a
#     character of solo representation of card
#
#     column: rank of the cards(in the order: 3456789TJQK2BR)
#     row: number of cards player has for a certain rank (0, 1, 2, 3, 4)
#
#     e.g.,
#     '6788889JQQKK222':
#     [[1 1 1 0 0 0 0 1 0 0 0 1 0 1 1]
#      [0 0 0 1 1 0 1 0 1 0 0 0 0 0 0]
#      [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0]
#      [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
#      [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]]
#
#      if player has '8888': in 6th column only 5th row is 1, rest are 0s. Note: 1st row of any rank of card is 1 iff
#      player does not have that card at all.
#     '''
#
#     if not cards:
#         return None
#     layer = 1
#     if len(cards) == 1:
#         rank = CARD_RANK_STR.index(cards[0])
#         plane[layer][rank] = 1
#         plane[0][rank] = 0
#     else:
#         for index, card in enumerate(cards):
#             if index == 0:
#                 continue
#             if card == cards[index - 1]:
#                 layer += 1
#             else:
#                 rank = CARD_RANK_STR.index(cards[index - 1])
#                 plane[layer][rank] = 1
#                 layer = 1
#                 plane[0][rank] = 0
#         rank = CARD_RANK_STR.index(cards[-1])
#         plane[layer][rank] = 1
#         plane[0][rank] = 0
#

def encode_cards(plane, cards):
    '''
    Encode cards and represent it into plane.

    Args:
        cards (list or str): list or str of cards, every entry is a
    character of solo representation of card

    column: rank of the cards(in the order: 3456789TJQK2BR)
    row: number of cards player has for a certain rank (1, 2, 3, 4)

    e.g.,
    '6788889JQQKK222':
     [0 0 0 1 1 0 1 0 1 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]]

     if player has '8888': in 6th column only 4th row is 1, rest rows are 0s.
    '''

    if not cards:
        return None
    layer = 0
    if len(cards) == 1:
        rank = CARD_RANK_STR.index(cards[0])
        plane[layer][rank] = 1
    else:
        for index, card in enumerate(cards):
            if index == 0:
                continue
            if card == cards[index - 1]:
                layer += 1
            else:
                rank = CARD_RANK_STR.index(cards[index - 1])
                plane[layer][rank] = 1
                layer = 0
        rank = CARD_RANK_STR.index(cards[-1])
        plane[layer][rank] = 1
    return plane

#
# def encode_cards_conv_5x15(plane, cards):
#     #### change the card encoding for convolutional layers ####
#     ''' Encode cards and represent it into plane.
#     Args:
#         cards (list or str): list or str of cards, every entry is a
#     character of solo representation of card
#
#     column: rank of the cards(in the order: 3456789TJQK2BR)
#     row: if the player has a certain number of cards for a certain rank (0, 1, 2, 3, 4)
#
#     e.g.,
#     '6788889JQQKK222':
#     [[1 1 1 0 0 0 0 1 0 0 0 1 0 1 1]
#      [0 0 0 1 1 1 1 0 1 1 1 0 1 0 0]
#      [0 0 0 0 0 1 0 0 0 1 1 0 1 0 0]
#      [0 0 0 0 0 1 0 0 0 0 0 0 1 0 0]
#      [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]]
#
#     if player has '8888': in 6th column 2-5th row are 1s, and 1st row is 0.
#     Note: 1st row of any rank of card is 1 iff player does not have that card at all.
#     '''
#
#     if not cards:
#         return None
#     layer = 1
#     if len(cards) == 1:
#         rank = CARD_RANK_STR.index(cards[0])
#         plane[layer][rank] = 1
#         plane[0][rank] = 0
#     else:
#         for index, card in enumerate(cards):
#             if index == 0:
#                 continue
#             if card == cards[index - 1]:
#                 layer += 1
#             else:
#                 rank = CARD_RANK_STR.index(cards[index - 1])
#                 for i in range(layer + 1):
#                     plane[i][rank] = 1
#                 layer = 1
#                 plane[0][rank] = 0
#         rank = CARD_RANK_STR.index(cards[-1])
#         for i in range(layer + 1):
#             plane[i][rank] = 1
#         plane[0][rank] = 0


def encode_cards_conv(plane, cards):
    #### change the card encoding for convolutional layers ####
    ''' Encode cards and represent it into plane.
    Args:
        cards (list or str): list or str of cards, every entry is a
    character of solo representation of card

    column: rank of the cards(in the order: 3456789TJQK2BR)
    row: if the player has a certain number of cards for a certain rank (0, 1, 2, 3, 4)

    e.g.,
    '6788889JQQKK222':
     [0 0 0 1 1 1 1 0 1 1 1 0 1 0 0]
     [0 0 0 0 0 1 0 0 0 1 1 0 1 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]]

    if player has '8888': in 6th column 2-5th row are 1s, and 1st row is 0.
    Note: 1st row of any rank of card is 1 iff player does not have that card at all.
    '''

    if not cards:
        return None
    layer = 0
    if len(cards) == 1:
        rank = CARD_RANK_STR.index(cards[0])
        plane[layer][rank] = 1
    else:
        for index, card in enumerate(cards):
            if index == 0:
                continue
            if card == cards[index - 1]:
                rank = CARD_RANK_STR.index(cards[index - 1])
                plane[layer][rank] = 1
                layer += 1
            else:
                rank = CARD_RANK_STR.index(cards[index - 1])
                plane[layer][rank] = 1
                layer = 0
        rank = CARD_RANK_STR.index(cards[-1])
        plane[layer][rank] = 1
    return plane

#
# def decode_cards_5x15(plane):
#     """
#     Decode cards from plane.
#     Get the string representation of cards from plane of state_obs.
#     :param
#     plane: state plane
#     :return:
#     (str) string representation of cards
#     """
#     plane = np.array(plane)
#     cards = ""
#
#     for i in range(15):
#         card_num = plane[:, i].tolist().index(1)
#
#         if card_num != 0:
#             for j in range(card_num):
#                 cards = cards + INDEX_CARD_RANK_STR[i]
#     return cards


def decode_card(plane):
    """
    Decode cards from plane.
    Get the string representation of cards from plane of state_obs.
    :param
    plane: state plane
    :return:
    (str) string representation of cards
    """
    plane = np.array(plane)
    cards = ""
    for i in range(15):
        if 1 not in plane[:, i]:
            card_num = 0
        else:
            card_num = plane[:, i].tolist().index(1) + 1

        if card_num != 0:
            for j in range(card_num):
                cards = cards + INDEX_CARD_RANK_STR[i]
    return cards

#
# def decode_cards_conv_5x15(plane):
#     '''
#     Decode cards from plane.
#     Get the string representation of cards from plane of state_obs.
#     :param
#     plane: state plane
#     :return:
#     (str) string representation of cards
#     '''
#     plane = np.array(plane)
#     cards = ""
#
#     for i in range(15):
#         card_num = 0
#         if not plane[0, i] == 1:
#             card_num = sum(plane[:, i].tolist())
#
#         if card_num != 0:
#             for j in range(card_num):
#                 cards = cards + INDEX_CARD_RANK_STR[i]
#     return cards


def decode_cards_conv(plane):
    """
    Decode cards from plane.
    Get the string representation of cards from plane of state_obs.
    :param
    plane: state plane
    :return:
    (str) string representation of cards
    """
    plane = np.array(plane)
    cards = ""

    for i in range(15):
        card_num = sum(plane[:, i].tolist())

        if card_num != 0:
            for j in range(card_num):
                cards = cards + INDEX_CARD_RANK_STR[i]
    return cards


def get_gt_cards(player, greater_player):
    '''
    Provide player's cards which are greater than the ones played by
    previous player in one round

    Args:
        player (DoudizhuPlayer object): the player waiting to play cards
        greater_player (DoudizhuPlayer object): the player who played current biggest cards.

    Returns:
        list: list of string of greater cards

    Note:
        1. return value contains 'pass'
    '''
    # add 'pass' to legal actions
    gt_cards = ['pass']
    current_hand = cards2str(player.current_hand)
    target_cards = greater_player.played_cards
    target_types = CARD_TYPE[0][target_cards]
    type_dict = {}
    for card_type, weight in target_types:
        if card_type not in type_dict:
            type_dict[card_type] = weight
    if 'rocket' in type_dict:
        return gt_cards
    type_dict['rocket'] = -1
    if 'bomb' not in type_dict:
        type_dict['bomb'] = -1
    for card_type, weight in type_dict.items():
        candidate = TYPE_CARD[card_type]
        for can_weight, cards_list in candidate.items():
            if int(can_weight) > int(weight):
                for cards in cards_list:
                    # TODO: improve efficiency
                    if cards not in gt_cards and contains_cards(current_hand, cards):
                        # if self.contains_cards(current_hand, cards):
                        gt_cards.append(cards)
    return gt_cards


def show_card(cards, info, n):
    if n == 1:
        print(info, cards)

    elif n == 2:
        if len(cards) == 0:
            return 0
        print(info)
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name + j.color)
            moves.append(names)
        print(moves)

    elif n == 3:
        print(info)
        names = []
        for i in cards:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            try:
                for j in i[1]:
                    tmp_name.append(j.name + j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            names.append(tmp)
        print(names)


def opt_legal(legal_action):
    """
       input: np.array of legal_action
       return: np.array of optimized legal_action
       optimize legal actions: if the card of one of the legal action is contained in another legal action,
       remove that action from legal_action
       e.g., np.array: [2, 3, 4, 9, 68, 76, 308] -> [9, 76, 308]
             string:   ['5', '6', '7', 'Q', '45678', '456789', 'pass'] -> ['Q', '456789', 'pass']
    """
    legal_action_str = [ACTION_ID_TO_STR[ac] for ac in legal_action]
    del_id = []
    new_legal_action = []
    for i in range(len(legal_action_str)):

        for j in range(i + 1, len(legal_action_str)):

            if legal_action_str[i] in legal_action_str[j] and '*' not in legal_action_str[i] \
                    and 'B' not in legal_action_str[i] and 'R' not in legal_action_str[i]:
                if legal_action_str[i] not in range(238, 269) and legal_action_str[i] not in range(281, 294):
                    del_id.append(legal_action[i])
                    break

    for card in legal_action:
        if card not in del_id:
            new_legal_action.append(card)

    return new_legal_action


def get_hand_length(state):
    # calculate the length of each player's current hand using trace
    # just for encoding the state, kind of a naive approach,
    # but we don't need to mess up with the state of different player or env.py

    landlord_hand_length = 0
    p1_hand_length = 0
    p2_hand_length = 0
    for action in (state['trace']):
        if action[0] == 0:
            if action[1] != 'pass':
                landlord_hand_length += len(action[1])
        elif action[0] == 1:
            if action[1] != 'pass':
                p1_hand_length += len(action[1])
        elif action[0] == 2:
            if action[1] != 'pass':
                p2_hand_length += len(action[1])

    # if the hand length is larger than 15, set it to 15 to match the shape of the state, since 15 or larger than
    # 15 won't really make a difference.
    landlord_hand_length = min((20 - landlord_hand_length), 15)
    p1_hand_length = min((17 - p1_hand_length), 15)
    p2_hand_length = min((17 - p2_hand_length), 15)
    return [landlord_hand_length, p1_hand_length, p2_hand_length]


def get_history(state):
    # get the played_cards(i.e., all cards that a player has played) of each player(0, 1, 2)
    # using trace from state_obs
    trace = state['trace']
    history = ['' for i in range(3)]
    played_cards = [[] for i in range(3)]
    for i in range(len(trace) - 1, -1, -1):
        _id, action = trace[i]
        if action == 'pass':
            continue
        played_cards[_id] += action
    for i in range(3):
        played_cards[i].sort(key=lambda x: CARD_RANK_STR_INDEX.get(x[0]))
        history[i] = history[i].join(played_cards[i])

    return history
