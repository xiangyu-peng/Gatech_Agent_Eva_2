import networkx as nx
import numpy as np
import openie
from fuzzywuzzy import fuzz
from jericho.util import clean

#check the rule change for dice
def check_rule_dice(graph_state, rule):
    if rule[-1] in graph_state['Dice'].keys():
        return True
    else:
        if graph_state['Dice']:
            return False
        else:
            graph_state.add_edge(rule[0], rule[2], rel=rule[1])
            return True

#define dice for rules
def dice_kg_rule(graph_state, line):
    left_bound = line.index(wake_up_words_rel['_die_'][0])
    right_bound = line.index(wake_up_words_rel['_die_'][1])
    dice = line[left_bound+1, right_bound]
    dice = map(lambda x : int(x), dice.split(','))
    sum_dice = sum(dice)
    #add sttributes to dice
    dice_att = 'dice = '+ str(sum_dice)
    if dice_att in graph_state.graph.keys():
        graph_state.graph[dice_att] += 1
    else:
        graph_state.graph[dice_att]  = 1
    #dice_num record
    rule = ['Dice', 'num', str(len(dice))]
    return check_rule_dice(graph_state, rule)

#define rules from location
#i.e.Go => move 8 steps => Vermont Avenue
def loc_kg_rule(graph_state, line):
    rule = [None,None,None]
    line = line.split(' ')
    index_rule = [i for i, val in enumerate(line) if val == wake_up_words_rel['_buy_']]
    rule[0] = line[index_rule[0] + 1]
    rule[2] = line[index_rule[1] + 1]
    rule[1] = 'move ' + str(dice_num) +' steps'
    return rule