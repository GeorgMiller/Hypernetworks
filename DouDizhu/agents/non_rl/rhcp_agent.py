"""
https://arxiv.org/pdf/1901.08925.pdf
heuristics-based: Recursive Handheld Cards Partitioning algorithm
The general idea of RHCP Algorithm is to take a best cards handing out strategy at each step.
It decides the hand to play purely by some hand- crafted hand value estimation function: pick a partitioning strategy
with the highest Strategy Score
"""
import copy
from collections import Counter
from doudizhu.card_mapping import card_mapping
from doudizhu.all_actions import ALL_ACTIONS

"""
Definitions of the category scores of all the legal categories.
MaxCard represents different values in different categories.
(e.g., the group '345678' belongs to the category Sequential Solos and its MaxCard is 8, the group 'QQQKKK89' belongs to 
the category Sequential Trios Series Take One, and its principal cards are QQQKKK thus its MaxCard is 13.).
Category                              Weight
None                                  0
Solo                                  MaxCard - 10
Pair                                  MaxCard - 10
Trio                                  MaxCard - 10
Sequential Solos                      MaxCard - 10 + 1
Sequential Pairs                      MaxCard - 10 + 1
Sequential Trios Take None            MaxCard - 10 + 1
Sequential Trios Take One             MaxCard - 10
Sequential Trios Take Two             MaxCard - 10
Sequential Trios Series Take One      (MaxCard - 3 + 1) / 2
Sequential Trios Series Take Two      (MaxCard - 3 + 1) / 2
Bomb                                  MaxCard - 3 + 7
Four Take Two Solos                   (MaxCard - 3) / 2
Four Take Two Pairs                   (MaxCard - 3) / 2
Nuke                                  20
"""
"""
handing out strategy: partitioning current hand cards H into two card groups: C and H\C, where C denotes the card group 
that is to be handed out and H\C denotes the card group that is to be kept as remained hand cards.
strategy score function that measures the quality of handing out strategy C given a set of hand cards H.
We select the best card group C* given current
hand cards H by, C* = arg maxC Q(C,H)
"""


# currently supported type of combination
# TRIO_SOLO_CHAIN, TRIO_PAIR_CHAIN(rl: 200-267), and FOUR_TWO_SOLO, FOUR_TWO_PAIR(rl: 268-293) are excluded for now
class COMB_TYPE:
    PASS, SOLO, PAIR, TRIO, TRIO_ONE, TRIO_TWO, SEQUENCE, SEQUENCE_TWO, SEQUENCE_THREE, BOMB, ROCKET = range(11)


def mapping(cards):
    # map card string to value of cards, used for calculating MaxCard value
    if len(cards) > 1:
        cards = cards[0]
    if '2' < cards <= '9':
        return int(cards)
    elif cards == 'T':
        return 10
    elif cards == 'J':
        return 11
    elif cards == 'Q':
        return 12
    elif cards == 'K':
        return 13
    elif cards == 'A':
        return 14
    elif cards == '2':
        return 15
    elif cards == 'B':
        return 16
    elif cards == 'R':
        return 17
    else:
        return 17


def comb1_gt_comb2(comb1, comb2):
    """
    determine if combination1 is greater than combination2
    args:
        comb1(str of cards), comb2(str of cards)
    return:
        True: if comb1 > comb2, else False
    """
    comb1 = card_mapping[comb1][0]
    comb2 = card_mapping[comb2][0]
    c1_type = comb1['type']
    c2_type = comb2['type']
    if c2_type == COMB_TYPE.PASS:
        return False
    if not comb1 or c1_type == COMB_TYPE.PASS:
        return True
    if c1_type == c2_type:
        cb1, cb2 = 0, 0
        if type(comb1['main']) != int:
            cb1 = mapping(comb1['main'])
        else:
            cb1 = comb1['main']
        if type(comb2['main']) != int:
            cb2 = mapping(comb2['main'])
        else:
            cb2 = comb2['main']
        if c1_type == COMB_TYPE.SEQUENCE:
            if cb1 != cb2:
                return False
            else:
                return mapping(comb2['sub']) > mapping(comb1['sub'])
        else:
            if cb1 == cb2 and c1_type != COMB_TYPE.SOLO and c1_type != COMB_TYPE.PAIR and c1_type != COMB_TYPE.TRIO:
                return cb2 > cb1
            else:
                return cb2 > cb1
    elif c2_type == COMB_TYPE.BOMB or c2_type == COMB_TYPE.ROCKET:
        return c2_type > c1_type
    return False


def get_greater_cards(last_cards, current_cards):
    """
    get the cards from current hand which are larger than the last played cards
    args:
        last_cards(str): string representation of cards last played)
        current_cards(str): string representation of cards from player's current hand
    return: a list of possible cards that are greater than last played cards
    """
    possible_gt_cards = []
    counter_cur_cards = Counter(current_cards)
    cur_cards_key = counter_cur_cards.keys()
    for comb in card_mapping.keys():
        counter_key = Counter(comb)
        is_combinable = True
        for k, v in counter_key.items():
            if not (k in cur_cards_key and v <= counter_cur_cards[k]):
                is_combinable = False
                break
        if is_combinable and comb1_gt_comb2(last_cards, comb):
            possible_gt_cards.append(comb)
    return possible_gt_cards


def remove_card(hand_cards, played_cards):
    # remove card being played from hand
    for i in played_cards:
        if i != 'pass':
            hand_cards = hand_cards.replace(i, '', 1)
    return hand_cards


def get_category_score(comb):
    """
    category scores of all the legal categories, based on max card
    """
    score = 0
    comb_type = card_mapping[comb][0]['type']
    if comb_type == COMB_TYPE.ROCKET:
        score = 20
    else:
        max_card = card_mapping[comb][0]['main']
        if type(max_card) != int:
            max_card = mapping(max_card)
        if comb_type == COMB_TYPE.PASS:
            score = 0
        elif comb_type in [COMB_TYPE.SOLO, COMB_TYPE.PAIR, COMB_TYPE.TRIO, COMB_TYPE.TRIO_ONE, COMB_TYPE.TRIO_TWO]:
            score = max_card - 10
        elif comb_type in [COMB_TYPE.SEQUENCE, COMB_TYPE.SEQUENCE_TWO]:
            score = max_card - 10 + 1
        elif comb_type == COMB_TYPE.SEQUENCE_THREE:
            score = (max_card - 3 + 1) / 2
        elif comb_type == COMB_TYPE.BOMB:
            score = max_card - 3 + 7
        else:
            print('invalid card type')
    return score


def get_all_legal_comb(card, cur_hands):
    result = []
    card_cnt = Counter(card)
    card_cnt_keys = card_cnt.keys()
    for hands in cur_hands:
        is_exist = True
        hand_cnt = Counter(hands)
        for key, value in hand_cnt.items():
            if key not in card_cnt_keys or value > card_cnt[key]:
                is_exist = False
                break
        if is_exist:
            result.append(hands)
    return result


def partition(legal_cards, cards, idx, possible_list, all_possible_list, level):
    length = len(legal_cards)
    if idx >= length:
        return
    if len(cards) == 0:
        all_possible_list.append(possible_list)
        return
    for i in range(idx, length):
        cur_card_cnt = Counter(cards)
        legal_cards_cnt = Counter(legal_cards[i])
        is_greater = True
        for key, value in legal_cards_cnt.items():
            if cur_card_cnt[key] < value:
                is_greater = False
                break
        if is_greater is True:
            tmp_card = remove_card(cards, legal_cards[i])
            tmp_possible_list = copy.deepcopy(possible_list)
            tmp_possible_list.append(legal_cards[i])
            partition(legal_cards, tmp_card, i, tmp_possible_list, all_possible_list, level+1)


def get_all_comb(legal_cards, cards):
    comb = []
    level = 0
    partition(legal_cards, cards, 0, [], comb, level)
    comb = sorted(comb, key=lambda i: (len(i), mapping(i[0])))
    result = []
    for cb in comb:
        if cb not in result:
            result.append(cb)
    return result


def decomposition(card_str):
    # get all comb of hand
    hands = ALL_ACTIONS
    # kernels out actions which have length bigger than 20 and sort all the actions
    hands = filter(lambda item: len(item) <= 20, hands)
    hands = sorted(hands, key=lambda i: len(i), reverse=True)
    # get all possible combination of cards
    all_legal_comb = get_all_legal_comb(card_str, hands)
    all_possible_comb = get_all_comb(all_legal_comb, card_str)
    return all_possible_comb


def get_partition_score(cards, is_pass=0):
    """
    calculate max score for all partition of handcard using category scores for each comb in the partition
    """
    max = -1000
    combs = decomposition(cards)
    #print(combs)
    for comb in combs:
        temp = 0
        for card in comb:
            temp += get_category_score(card)
        if max < temp - 7 * (len(comb) + 1):
            max = temp - 7 * (len(comb) + 1)

    return max


def following_round(last_cards, hand_cards):
    # rule of following the round
    playable_cards = get_greater_cards(last_cards, hand_cards)
    # if the all the cards in hand is a legal_action, play all the cards at once
    if hand_cards in playable_cards:
        return hand_cards
    if 'B' in hand_cards and 'R' in hand_cards:
        hand_card_cp = copy.deepcopy(hand_cards)
        hand_card_cp = remove_card(hand_card_cp, 'B')
        hand_card_cp = remove_card(hand_card_cp, 'R')
        if hand_card_cp in ALL_ACTIONS:
            return 'BR'
    same_type = []
    bomb_type = []
    last_cards_type = card_mapping[last_cards][0]['type']
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == last_cards_type:
            same_type.append(cards)
        if card_mapping[cards][0]['type'] in [COMB_TYPE.ROCKET, COMB_TYPE.BOMB]:
            bomb_type.append(cards)

    # play the same type of cards
    if_pass_score = get_partition_score(hand_cards, 1)
    score_dict = {}
    max_card_key = ''
    same_type = sorted(same_type, key=lambda x: x[0])
    for cards in same_type:
        hand_cards_cp = copy.deepcopy(hand_cards)
        hand_cards_cp = remove_card(hand_cards_cp, cards)
        score = get_partition_score(hand_cards_cp, 0)
        if max_card_key == '' or score_dict[max_card_key] < score:
            max_card_key = cards
        if cards not in score_dict.keys():
            score_dict[cards] = score
        else:
            if score_dict[cards] < score:
                score_dict[cards] = score
    if max_card_key in score_dict.keys() and score_dict[max_card_key] > (if_pass_score - 5):
        return max_card_key
    # play bomb or rocket
    bomb_dict = {}
    max_card_key = ''
    bomb_type = sorted(bomb_type, key=lambda x: x[0])
    for cards in bomb_type:
        hand_cards_cp = copy.deepcopy(hand_cards)
        hand_cards_cp = remove_card(hand_cards_cp, cards)
        score = get_partition_score(hand_cards_cp, 0)
        if max_card_key == '' or bomb_dict[max_card_key] < score:
            max_card_key = cards
        if cards not in bomb_dict.keys():
            bomb_dict[cards] = score
        else:
            if bomb_dict[cards] < score:
                bomb_dict[cards] = score
    if max_card_key in bomb_dict.keys() and bomb_dict[max_card_key] > (if_pass_score - 5):
        return max_card_key
    # pass
    return 'pass'


def leading_round(hand_cards):
    # rule of leading the round(i.e., the first player to play a card in one round)
    all_cards = ALL_ACTIONS
    # if it's possible to play all the cards in hand at once
    if hand_cards in all_cards:
        return hand_cards
    if 'B' in hand_cards and 'R' in hand_cards:
        hand_card_cp = copy.deepcopy(hand_cards)
        hand_card_cp = remove_card(hand_card_cp, 'B')
        hand_card_cp = remove_card(hand_card_cp, 'R')
        if hand_card_cp in ALL_ACTIONS:
            return 'BR'

    playable_cards = []
    hand_card_cnt = Counter(hand_cards)
    hand_card_cnt_keys = hand_card_cnt.keys()
    for card in all_cards:
        is_exist = True
        tmp_counter = Counter(card)
        for key, value in tmp_counter.items():
            if key not in hand_card_cnt_keys or value > hand_card_cnt[key]:
                is_exist = False
                break
        if is_exist:
            playable_cards.append(card)

    # TODO: Think more about the order of comb to play.
    # the order of combs to play makes a big difference.
    trio_one_list, trio_two_list = [], []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.TRIO_ONE:
            trio_one_list.append(cards)
        elif card_mapping[cards][0]['type'] == COMB_TYPE.TRIO_TWO:
            trio_two_list.append(cards)

    # play trio_one
    if len(trio_one_list) != 0:
        trio_one_dict = {}
        max_card_key = ''
        for t_1 in trio_one_list:
            hand_cards_cp = copy.deepcopy(hand_cards)
            hand_cards_cp = remove_card(hand_cards_cp, t_1)
            score = get_partition_score(hand_cards_cp, 0)
            if max_card_key == '':
                max_card_key = t_1
                trio_one_dict[t_1] = score
            elif t_1 in trio_one_dict.keys():
                if trio_one_dict[t_1] < score:
                    trio_one_dict[t_1] = score
            else:
                trio_one_dict[t_1] = score
                if trio_one_dict[max_card_key] < score:
                    max_card_key = t_1
        return max_card_key

    # play trio_two
    if len(trio_two_list) != 0:
        trio_two_dict = {}
        max_card_key = ''
        for t_2 in trio_two_list:
            hand_cards_cp = copy.deepcopy(hand_cards)
            hand_cards_cp = remove_card(hand_cards_cp, t_2)
            score = get_partition_score(hand_cards_cp, 0)
            if max_card_key == '':
                max_card_key = t_2
                trio_two_dict[t_2] = score
            elif t_2 in trio_two_dict.keys():
                if trio_two_dict[t_2] < score:
                    trio_two_dict[t_2] = score
            else:
                trio_two_dict[t_2] = score
                if trio_two_dict[max_card_key] < score:
                    max_card_key = t_2
        return max_card_key

    # play sequence_two
    sequence_two = []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.SEQUENCE_TWO:
            sequence_two.append(cards)
    if len(sequence_two) != 0:
        return sequence_two[0]

    # play sequence_three
    sequence_three = []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.SEQUENCE_THREE:
            sequence_three.append(cards)
    if len(sequence_three) != 0:
        return sequence_three[0]

    # play sequence
    sequence = []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.SEQUENCE:
            sequence.append(cards)
    if len(sequence) != 0:
        return sequence[0]

    # play trio
    trio = []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.TRIO:
            trio.append(cards)
    if len(trio) != 0:
        return trio[0]

    # play solo
    solo = []
    for cards in playable_cards:
        if len(cards) == 1 and hand_card_cnt != 4:
            solo.append(cards)
    if len(solo) != 0:
        return solo[0]

    # play pair
    pair = []
    for cards in playable_cards:
        if len(cards) == 2 and hand_card_cnt != 4:
            pair.append(cards)
    if len(pair) != 0:
        return pair[0]

    # play bomb
    bomb = []
    for cards in playable_cards:
        if card_mapping[cards][0]['type'] == COMB_TYPE.BOMB:
            bomb.append(cards)
    if len(bomb) != 0:
        return bomb[0]


class RHCPAgent(object):
    '''
    Dou Dizhu Rule agent version 2 - RHCPAgent
    '''

    def __init__(self, action_num):
        self.use_raw = True
        self.action_num = action_num

    def step(self, state):
        '''
        Predict the action given raw state.
        Args:
            state (dict): Raw state from the doudizhu

        Returns:
            action (str): Predicted action
        '''

        # get the string representation for current hand and trace using state raw observation dict.
        state = state['raw_obs']
        trace = state['trace']
        current_hand = state['current_hand']
        print(current_hand)

        # if the player is the first one to play an action in one round
        if len(trace) == 0 or (len(trace) >= 3 and trace[-1][1] == 'pass' and trace[-2][1] == 'pass'):
            action = leading_round(current_hand)
            print(action)

        # if the player is following the round
        else:
            last_cards = trace[-1][1]
            if trace[-1][1] == 'pass':
                last_cards = trace[-2][1]
            action = following_round(last_cards, current_hand)
            print(action)
        # action = card_mapping[action][0]['rl']
        return action

    def eval_step(self, state):
        '''
        Step for evaluation. The same to step
        '''
        return self.step(state), []
