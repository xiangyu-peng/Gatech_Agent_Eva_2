import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/GNOME-p3','')
upper_path_eva = upper_path + '/Evaluation/GNOME-p3'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation/GNOME-p3')
#####################################
from monopoly_simulator.card import *
from collections import Counter

def detect_card_nevelty(current_gameboard, gameboard_ini):
    """
    Detect card type and number change with comparing gameboard
    """
    novelty = []
    card_keys = ['chance_cards', 'community_chest_cards']
    for card_key in card_keys:
        if card_type_detect(current_gameboard[card_key]) != card_type_detect(gameboard_ini[card_key]):
            print('different')
            novelty = dict_difference(card_type_detect(gameboard_ini[card_key]), card_type_detect(current_gameboard[card_key]))
        else:
            print('same')
    if novelty:
        return novelty
    else:
        return None


def detect_contingent(current_gameboard, gameboard_ini):
    """
    Detect the contingent change in gameboards
    """
    novelty = []
    if current_gameboard['bank'].mortgage_percentage != gameboard_ini['bank'].mortgage_percentage:
        novelty.append(['bank percentage', gameboard_ini['bank'].mortgage_percentage, current_gameboard['bank'].mortgage_percentage])
    if novelty:
        return novelty
    else:
        return None

def dict_difference(dict1, dict2):
    diff = []
    for key in dict1:
        if key in dict2:
            if dict1[key] != dict2[key]:
                diff.append([key, dict1[key], dict2[key]])
        else:
            diff.append([key, dict1[key], None])

    for key in dict2:
        if key not in dict1:
            diff.append([key, None, dict2[key]])
    return diff

def card_move(card):
    if type(card) == MovementCard:
        return card.destination.name
    elif type(card) == MovementRelativeCard:
        return card.new_relative_position
    elif type(card) == CashFromBankCard:
        return card.amount
    elif type(card) == ContingentCashFromBankCard:
        print(card.contingency)
        return card.contingency
    elif type(card) == CashFromPlayersCard:
        return card.amount_per_player
    else:
        return None

def card_type_detect(cards):
    card_tuple = dict()  # (type, num, move)
    for card in cards:
        if card.name in card_tuple:
            card_tuple[card.name][1] += 1
        else:
            card_tuple[card.name] = [card.card_type, 1, None]
        card_tuple[card.name][2] = card_move(card)
    return card_tuple