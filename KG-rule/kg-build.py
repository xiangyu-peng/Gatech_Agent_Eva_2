import networkx as nx
import numpy as np
import openie
from fuzzywuzzy import fuzz
from jericho.util import clean
import matplotlib.pyplot as plt
from kg_build_func import *
import pygraphviz as PG


# graph_state = nx.DiGraph()
# rule = ['a', 'like','b']
# graph_state.add_edge(rule[0], rule[2], rel=rule[1])
# rule = ['a', 'love','c']
# graph_state.add_edge(rule[0], rule[2], rel=rule[1])
# rule = ['c', 'like','d']
# graph_state.add_edge(rule[0], rule[2], rel=rule[1])
# graph_state.graph['a'] = 'boy'
# visualize(graph_state)


class KGEnv(object):
    def __init__(self):
        self.graph_state = nx.DiGraph()
        self.wake_up_words = {'-buy-', '-die-', '-pay-', '-loc-', '-inc-'}
        self.wake_up_words_rel = {'-die-':['[', ']'], '-loc-': ['position','and'], '-buy-': ['asset', 'amount','buy from bank'],
                                  '-pay-': ['in', 'be', 'with']}
        self.wake_up_words_func = {'-die-': self.dice_kg_rule, '-loc-': self.loc_kg_rule, '-buy-': self.buy_kg_rule, '-pay-':self.pay_kg_rule,
                                   '-inc-': inc_kg_rule}
        self.dice_sum = 0


    def visualize(self):
        # import matplotlib.pyplot as plt
        pos = nx.nx_agraph.graphviz_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        print(edge_labels)
        f = plt.figure()
        #need a better way to set the edge width
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels, font_size=2, label_pos=0.5, width=0.01)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=200, font_size=2)
        f.savefig("graph.png", dpi=500)

    #add edges to graph
    def add_edge(self,rule):
        self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])

    def get_node_with_rel(self, node, rel):
        for node, node_out in self.graph_state.edges([node]):
            if rel == self.graph_state[node][node_out]['rel']:
                break
        return out_node

    # check the rule change for dice and record the #of dice we use
    def dice_rule_check(self, rule):
        if 'Dice' in self.graph_state.nodes:
            if rule[-1] in self.graph_state['Dice'].keys():
                # if 'Dice' is node already and # of dice does not change, we do nothing
                return True
            else:
                if self.graph_state['Dice']:

                    self.graph_state.remove_edge('Dice', get_node_with_rel('Dice', rule[1]))
                    self.add_edge(rule)
                    return False
        self.add_edge(rule)
        return True


    # define dice for rules: in gameplay.py
    def dice_kg_rule(self, line):
        left_bound = line.index(self.wake_up_words_rel['-die-'][0]) + 1
        right_bound = line.index(self.wake_up_words_rel['-die-'][1])
        dice = line[left_bound : right_bound]
        dice = [int(x) for x in dice.split(', ')]
        self.dice_sum = sum(dice)

        # add attributes to dice => memory
        dice_att = 'Dice = ' + str(self.dice_sum)
        if dice_att in self.graph_state.graph.keys():
            self.graph_state.graph[dice_att] += 1
        else:
            self.graph_state.graph[dice_att] = 1

        # dice_num record
        rule = ['Dice', 'num', str(len(dice))]
        return self.dice_rule_check(rule)



    #helper fuction for buy, to detect the rule change of buy and also adds the rule of buy

    def buy_loc_rule_check(self, rule):
        if rule[0] in self.graph_state.nodes:
            for en, rel in self.graph_state[rule[0]].items():
                if rel['rel'] == rule[1]:
                    return True if en == rule[2] else False
        self.add_edge(rule)
        return True

    #location.py
    # define rules from location
    # i.e.Go => move 8 steps => Vermont Avenue
    def loc_kg_rule(self, line):
        rule = [None, None, None]
        line = line.split(' ')
        index_rule = [i for i, val in enumerate(line) if val == self.wake_up_words_rel['-loc-'][0]]
        print()
        rule[0] = ' '.join(line[index_rule[0] + 1 : line.index(self.wake_up_words_rel['-loc-'][1])])
        rule[2] = ' '.join(line[index_rule[1] + 1: ])
        rule[1] = 'move ' + str(self.dice_sum) + ' steps'
        return self.buy_loc_rule_check(rule)

    # in actions_choices.py
    def buy_kg_rule(self, line):
        wake_up_word = line[:5]
        line = line.split(' ')
        rule = [None, self.wake_up_words_rel[wake_up_word][2], None]
        rule[0] = ' '.join(line[line.index(self.wake_up_words_rel['-buy-'][0]) + 1 :])
        rule[2] = line[line.index(self.wake_up_words_rel['-buy-'][1]) + 1]
        return self.buy_loc_rule_check(rule)

    def pay_kg_rule(self,line):
        line = line.split(' ')
        rule = [None, 'pay rent with ', None]
        rule[0] = ' '.join(line[line.index(self.wake_up_words_rel['-pay-'][0]) + 1:])
        rule[2] = line[line.index(self.wake_up_words_rel['-pay-'][1]) + 1]
        rule[1] += line[line.index(self.wake_up_words_rel['-pay-'][2]) + 1]
        rule[1] += ' houses'
        return self.buy_loc_rule_check(rule)

    #in card_utilities.py check for go '-inc-'
    def inc_kg_rule(self,line):
        rule = ['Dice', 'passes', None]
        rule[-1] = line[-1]
        return self.buy_loc_rule_check(rule)

    #record color of spaces
    def col_kg_rule(self,line):
    def update(self, file='/media/becky/GNOME/monopoly_simulator/game-1-log.txt'):
        log_file = open(file,'r')
        try:
            for line in log_file:
                 if line[:5] in self.wake_up_words:
                     line = line.replace('  ', ' ')
                     line = line.replace('   ', ' ')
                     line = line.rstrip()
                     print(line)
                     self.wake_up_words_func[line[:5]](line)

        finally:
             log_file.close()
        self.visualize()

G = KGEnv()
G.update()