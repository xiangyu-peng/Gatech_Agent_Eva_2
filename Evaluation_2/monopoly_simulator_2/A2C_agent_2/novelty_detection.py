import os, sys
upper_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(upper_path)

import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr
from zipfile import ZipFile
from configparser import ConfigParser
import wget
import numpy as np
from A2C_agent_2.interface_eva import Interface_eva
from scipy.sparse import csr_matrix, load_npz, save_npz
from configparser import ConfigParser
from collections import Counter
import random
from scipy import stats
import pickle
from monopoly_simulator.gameplay import set_up_board
from monopoly_simulator.agent import Agent
import copy
import json

from stanfordnlp.server import CoreNLPClient

class History_Record(object):
    def params_read(self, config_data, keys):
        '''
        Read config.ini file
        :param config_data:
        :param keys (string): sections in config file
        :return: a dict with info in config file
        '''
        params = {}
        for key in config_data[keys]:
            v = eval(config_data[keys][key])
            params[key] = v
        return params

    def save_json(self, save_dict, save_path):
        """
        Save kg dict to json file
        For kg save_dict = self.kg_rel
        For kg save_path = self.jsonfile
        :param level:
        :return: None
        """
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def read_json(self, load_path):
        """
        Read kg dict file from json file
        :param level:
        :return: None
        """
        load_dict = dict()
        with open(load_path, 'r')as f:
            load_dict = json.load(f)

        return load_dict


# NOvelty Detection class
# class KG_OpenIE_eva(History_Record):
#     # def __init__(self, gameboard=None,\
#     #              core_nlp_version: str = '2018-10-05', config_file=upper_path + '/A2C_agent/config.ini'):
#     def __init__(self, gameboard,
#                  matrix_file_name,
#                  entity_file_name,
#                  core_nlp_version: str = '2018-10-05',
#                  config_file=None):
#         self.upper_path = upper_path
#         # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
#
#         # nlp server env
#         self.remote_url = 'https://nlp.stanford.edu/software/stanford-corenlp-full-{}.zip'.format(core_nlp_version)
#         self.install_dir = Path('~/.stanfordnlp_resources/').expanduser()
#         self.install_dir.mkdir(exist_ok=True)
#         if not (self.install_dir / Path('stanford-corenlp-full-{}'.format(core_nlp_version))).exists():
#             print('Downloading from %s.' % self.remote_url)
#             output_filename = wget.download(self.remote_url, out=str(self.install_dir))
#             print('\nExtracting to %s.' % self.install_dir)
#             zf = ZipFile(output_filename)
#             zf.extractall(path=self.install_dir)
#             zf.close()
#         os.environ['CORENLP_HOME'] = str(self.install_dir / 'stanford-corenlp-full-2018-10-05')
#         self.client = CoreNLPClient(annotators=['openie'], memory='8G')
#
#         config_data = ConfigParser()
#         config_data.read(config_file)
#         self.params = self.params_read(config_data, keys='kg')
#         self.jsonfile = self.upper_path + self.params['jsonfile']
#         self.relations = ['price', 'rent', 'located', 'colored', 'classified', 'away', 'type', 'cost', 'direct',
#                           'mortgaged', 'increment']
#         self.relations_matrix = ['is priced at', 'is price-1-house at', 'is rented-0-house at', 'is rented-1-house at', \
#                                  'is rented-2-house at', 'is rented-3-house at', 'is rented-4-house at',
#                                  'is rented-1-hotel at']  # , 'is colored as', 'is classified as']
#         self.relations_node = ['Price_', 'Price_1_house_', 'Rent_0_house_', 'Rent_1_house_', 'Rent_2_house_',
#                                'Rent_3_house_', \
#                                'Rent_4_house_', 'Rent_1_hotel_']
#         self.under_list = []
#         self.upper_list = []
#         #####################
#         self.price_list_under = [0, 100, 150, 200, 300, 400, 1000]
#         self.under_list.append(self.price_list_under)
#         self.price_list_upper = [99, 149, 199, 299, 399, 999, sys.maxsize]
#         self.upper_list.append(self.price_list_upper)
#         #####################
#         self.price_1_house_list_under = [0, 50, 100, 150, 200]
#         self.under_list.append(self.price_1_house_list_under)
#         self.price_1_house_list_upper = [49, 99, 149, 199, sys.maxsize]
#         self.upper_list.append(self.price_1_house_list_upper)
#         #####################
#         self.rent_0_house_list_under = [0, 5, 10, 15, 20, 30, 50]
#         self.under_list.append(self.rent_0_house_list_under)
#         self.rent_0_house_list_upper = [4, 9, 14, 19, 29, 49, sys.maxsize]
#         self.upper_list.append(self.rent_0_house_list_upper)
#         #####################
#         self.rent_1_house_list_under = [0, 50, 100, 150, 200]
#         self.under_list.append(self.rent_1_house_list_under)
#         self.rent_1_house_list_upper = [49, 99, 149, 199, sys.maxsize]
#         self.upper_list.append(self.rent_1_house_list_upper)
#         #####################
#         self.rent_2_house_list_under = [0, 50, 100, 200, 300, 400, 500, 600]
#         self.under_list.append(self.rent_2_house_list_under)
#         self.rent_2_house_list_upper = [49, 99, 199, 299, 399, 499, 599, sys.maxsize]
#         self.upper_list.append(self.rent_2_house_list_upper)
#         #####################
#         self.rent_3_house_list_under = [0, 100, 300, 500, 800, 1000, 1200, 1400]
#         self.under_list.append(self.rent_3_house_list_under)
#         self.rent_3_house_list_upper = [99, 299, 499, 799, 999, 1199, 1399, sys.maxsize]
#         self.upper_list.append(self.rent_3_house_list_upper)
#         #####################
#         self.rent_4_house_list_under = [0, 200, 500, 800, 1200, 1700]
#         self.under_list.append(self.rent_4_house_list_under)
#         self.rent_4_house_list_upper = [199, 499, 799, 1199, 1699, sys.maxsize]
#         self.upper_list.append(self.rent_4_house_list_upper)
#         #####################
#         self.rent_1_hotel_list_under = [0, 300, 600, 1000, 1500, 2000]
#         self.under_list.append(self.rent_1_hotel_list_under)
#         self.rent_1_hotel_list_upper = [299, 599, 999, 1499, 1999, sys.maxsize]
#         self.upper_list.append(self.rent_1_hotel_list_upper)
#         #####################
#
#         self.kg_rel = dict()  # the total kg rule for "rel" KG
#         self.kg_sub = dict()  # the total kg rule for "sub" KG
#         self.kg_set = set()  # the set recording the kg rule for searching if the rule exists quickly
#         self.kg_rel_diff = dict()  # the dict() to record the rule change
#         self.kg_sub_diff = dict()
#         self.kg_change_bool = False
#
#         self.kg_introduced = False
#         self.new_kg_tuple = dict()
#         self.update_num = 1
#         self.update_interval = self.params['update_interval']
#         self.detection_num = self.params['detection_num']
#         self.kg_change = []
#         self.history_update_interval = self.params['history_update_interval']
#         self.location_record = dict()
#
#         # For kg to matrix
#         self.matrix_params = self.params_read(config_data, keys='matrix')
#         self.entity_num = self.matrix_params['entity_num']
#         self.action_num = self.matrix_params['action_num']
#         self.sparse_matrix = []
#         self.action_name = ['is ' + str(i) + '-step away from' for i in range(1, 41)]
#         self.board_name = []
#
#         self.set_gameboard(gameboard)  # get info about the gameboard each round
#         # self.board_name = ['Go','Mediterranean-Avenue', 'Community Chest-One',
#         #         'Baltic-Avenue', 'Income Tax', 'Reading Railroad', 'Oriental-Avenue',
#         #         'Chance-One', 'Vermont-Avenue', 'Connecticut-Avenue', 'In-Jail/Just-Visiting',
#         #         'St. Charles Place', 'Electric Company', 'States-Avenue', 'Virginia-Avenue',
#         #         'Pennsylvania Railroad', 'St. James Place', 'Community Chest-Two', 'Tennessee-Avenue',
#         #         'New-York-Avenue', 'Free Parking', 'Kentucky-Avenue', 'Chance-Two', 'Indiana-Avenue',
#         #         'Illinois-Avenue', 'B&O Railroad', 'Atlantic-Avenue', 'Ventnor-Avenue',
#         #         'Water Works', 'Marvin Gardens', 'Go-to-Jail', 'Pacific-Avenue', 'North-Carolina-Avenue',
#         #         'Community Chest-Three', 'Pennsylvania-Avenue', 'Short Line', 'Chance-Three', 'Park Place',
#         #                                 'Luxury Tax', 'Boardwalk']
#         self.node_number = 0
#         self.sparse_matrix_dict = self.build_empty_matrix_dict()
#         self.matrix_folder = self.upper_path + self.matrix_params['matrix_folder']
#         self.matrix_file_path = self.matrix_folder + matrix_file_name
#         self.entity_file_path = self.matrix_folder + entity_file_name
#         self.kg_vector = np.zeros([len(self.relations_matrix), len(self.board_name)])
#         self.vector_file = self.upper_path + self.matrix_params['vector_file']
#
#         # Dice Novelty
#         self.dice = Novelty_Detection_Dice(config_file)
#         self.text_dice_num = 0
#         self.dice_novelty = []
#         # self.text_card_num = 0
#         # self.card = Novelty_Detection_Card(config_file)
#
#     def set_gameboard(self, gameboard):
#         if gameboard:
#             self.gameboard = gameboard
#             self.board_name = gameboard['location_sequence'].copy()
#             for i, name in enumerate(self.board_name):
#                 self.board_name[i] = '-'.join(name.split(' '))
#
#         # def build_empty_matrix_dict(self):
#         #     sparse_matrix_dict = dict()
#         #     for rel in self.action_name:
#         #         sparse_matrix_dict[rel] = dict()
#         #         sparse_matrix_dict[rel]['row'] = []
#         #         sparse_matrix_dict[rel]['col'] = []
#         #         sparse_matrix_dict[rel]['data'] = []
#         #     return sparse_matrix_dict
#
#     def hash_money(self, money, index):
#         under_list = self.under_list[index]
#         upper_list = self.upper_list[index]
#         for i, under_limit in enumerate(under_list):
#             if money >= under_limit and money <= upper_list[i]:
#                 return under_limit
#
#         return -1
#
#     def build_empty_matrix_dict(self):
#         """
#         Build a empty dict for storing the matrix
#         :return: a dict
#         """
#         sparse_matrix_dict = dict()
#         sparse_matrix_dict['number_nodes'] = dict()
#         sparse_matrix_dict['nodes_number'] = dict()
#         sparse_matrix_dict['out'] = dict()
#         sparse_matrix_dict['in'] = dict()
#         sparse_matrix_dict['number_rel'] = dict()
#         sparse_matrix_dict['rel_number'] = dict()
#         sparse_matrix_dict['all_rel'] = dict()
#
#         # Define the relation/edge number
#         for i, rel in enumerate(self.relations_matrix):
#             sparse_matrix_dict['number_rel'][i] = rel
#             sparse_matrix_dict['rel_number'][rel] = i
#
#         # Define the number of nodes
#         # name of location
#
#         for i, node in enumerate(self.board_name):
#             sparse_matrix_dict['number_nodes'][i] = node
#             if node in sparse_matrix_dict['nodes_number']:
#                 id_value = sparse_matrix_dict['nodes_number'][node]
#                 if type(id_value) == list:
#                     id_value.append(i)
#                 else:
#                     id_value = [id_value] + [i]
#                 sparse_matrix_dict['nodes_number'][node] = id_value
#             else:
#                 sparse_matrix_dict['nodes_number'][node] = i
#         index_now = i + 1
#
#         # [7]Price: 0-99; 100-149;150-199;200-299;300-399;400-999; 1000+
#         for i, price in enumerate(self.price_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Price_' + str(price)
#             sparse_matrix_dict['nodes_number']['Price_' + str(price)] = i + index_now
#         index_now += i + 1
#
#         # [5]Price: 1 - house: 0-49; 50-99; 100-149; 150-199; 200+
#         for i, price in enumerate(self.price_1_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Price_1_house_' + str(price)
#             sparse_matrix_dict['nodes_number']['Price_1_house_' + str(price)] = i + index_now
#         index_now += i + 1
#         # #location
#         # for k, loc in enumerate(range(40)):
#         #     sparse_matrix_dict['number_nodes'][i + j + k + 2] = 'Location_' + str(k)
#         #     sparse_matrix_dict['nodes_number']['Location_' + str(k)] = i + j + k + 2
#
#         # [7] Rent_0_house: 0-4, 5-9, 10-14, 15-19, 20-29, 30-49, 50+
#         for i, rent in enumerate(self.rent_0_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_0_house_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_0_house_' + str(rent)] = i + index_now
#         index_now += i + 1  # update the index for next node
#
#         # [5] Rent_1_house: 0-49, 50-99, 100-149, 150-199, 200+
#         for i, rent in enumerate(self.rent_1_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_1_house_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_1_house_' + str(rent)] = i + index_now
#         index_now += i + 1  # update the index for next node
#
#         # [8] Rent_2_house: 0-49, 50-99, 100-199, 200-299, 300-399,400-499,500-599,600+
#         for i, rent in enumerate(self.rent_2_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_2_house_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_2_house_' + str(rent)] = i + index_now
#         index_now += i + 1  # update the index for next node
#
#         # [8] Rent_3_house: 0-99, 100-299, 300-499, 500-799, 800-999,1000-1199,1200-1399,1400+
#         for i, rent in enumerate(self.rent_3_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_3_house_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_3_house_' + str(rent)] = i + index_now
#         index_now += i + 1  # update the index for next node
#
#         # [6] Rent_4_house: 0-199, 200-499, 500-799, 800-1199,1200-1699,1700+
#         for i, rent in enumerate(self.rent_4_house_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_4_house_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_4_house_' + str(rent)] = i + index_now
#         index_now += i + 1
#
#         # [6] Rent_1_hotel: 0-299, 300-599, 600-999, 1000-1499,1500-1999,2000+
#         for i, rent in enumerate(self.rent_1_hotel_list_under):
#             sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_1_hotel_' + str(rent)
#             sparse_matrix_dict['nodes_number']['Rent_1_hotel_' + str(rent)] = i + index_now
#
#         # # Define 'in' column names
#         # for rel in self.relations_matrix:
#         #     sparse_matrix_dict['in'][rel] = dict()
#         #     for node in range(self.node_number):
#         #         sparse_matrix_dict['in'][rel][node] = None
#
#         self.node_number = len(list(sparse_matrix_dict['number_nodes'].keys()))
#
#         # Define 'out' column names
#         for rel in self.relations_matrix:
#             sparse_matrix_dict['out'][rel] = dict()
#             for node in range(self.node_number):
#                 sparse_matrix_dict['out'][rel][node] = None
#
#         # for rel in self.action_name:
#         #     sparse_matrix_dict[rel] = dict()
#         #     sparse_matrix_dict[rel]['row'] = []
#         #     sparse_matrix_dict[rel]['col'] = []
#         #     sparse_matrix_dict[rel]['data'] = []
#         return sparse_matrix_dict
#
#     def annotate(self, text: str, properties_key: str = None, properties: dict = None, simple_format: bool = True):
#         """
#         Annotate text to triples: sub, rel, obj
#         :param (str | unicode) text: raw text for the CoreNLPServer to parse
#         :param (str) properties_key: key into properties cache for the client
#         :param (dict) properties: additional request properties (written on top of defaults)
#         :param (bool) simple_format: whether to return the full format of CoreNLP or a simple dict.
#         :return: Depending on simple_format: full or simpler format of triples <subject, relation, object>.
#         """
#         # https://stanfordnlp.github.io/CoreNLP/openie.html
#
#         text = text.replace('_',
#                             '-')  # Some words in logger info containiis not detected in openie, hence we replace '_' to '-'
#
#         core_nlp_output = self.client.annotate(text=text, annotators=['openie'], output_format='json',
#                                                properties_key=properties_key, properties=properties)
#         if simple_format:
#             triples = []
#             for sentence in core_nlp_output['sentences']:
#                 for triple in sentence['openie']:
#
#                     for rel in self.relations:
#                         length_sub = len(triple['subject'].split(' '))
#                         length_obj = len(triple['object'].split(' '))
#                         if rel in triple['relation'] and length_sub == 1 and length_obj == 1 and \
#                                 triple['subject'] in self.board_name:
#
#                             # map B&0-Railroad
#                             if 'B-' in triple['subject']:
#                                 triple['subject'] = 'B&O-Railroad'
#                             if 'B-' in triple['object']:
#                                 triple['object'] = 'B&O-Railroad'
#                             if triple['subject'] == 'GO':
#                                 triple['subject'] = 'Go'
#                             triples.append({
#                                 'subject': triple['subject'],
#                                 'relation': triple['relation'],
#                                 'object': triple['object']
#                             })
#
#                     # triples.append({
#                     #     'subject': triple['subject'],
#                     #     'relation': triple['relation'],
#                     #     'object': triple['object']
#                     # })
#             return triples
#         else:
#             return core_nlp_output
#
#     def kg_update(self, triple, level='sub', type=None):
#         """
#         After detecting rule change, update kg and also return the diff of kg
#         * When adding the game rule, if we see contradiction, we will call ***self.kg_update()*** to update the knowledge
#           graph and return difference. PS: we did not update the game rule when there is contradiction in *self.kg_add()*.
#                     * Update **self.kg_rel_diff** to record the difference.
#                     * Update knowledge graph, **self.kg_rel**
#                     * Call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
#                     * return the difference: [sub, rel, old-obj, new-obj]
#
#         :param triple (dict): triple is a dict with three keys: subject, relation and object
#         :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
#         :return: A tuple (sub, rel, diff)
#         """
#         if level == 'sub':
#             if triple['subject'] not in self.kg_sub_diff.keys():
#                 self.kg_sub_diff[triple['subject']] = dict()
#                 self.kg_sub_diff[triple['subject']][triple['relation']] = [
#                     self.kg_sub[triple['subject']][triple['relation']]]
#             self.kg_sub[triple['subject']][triple['relation']] = triple['object']
#             self.kg_sub_diff[triple['subject']][triple['relation']].append(triple['object'])
#             return [triple['subject'], triple['relation'], self.kg_sub_diff[triple['subject']][triple['relation']]]
#
#         else:
#             if triple['relation'] not in self.kg_rel_diff.keys():
#                 self.kg_rel_diff[triple['relation']] = dict()
#
#             if triple['subject'] not in self.kg_rel_diff[triple['relation']].keys():
#                 if type == 'change':
#                     self.kg_rel_diff[triple['relation']][triple['subject']] = [
#                         self.kg_rel[triple['relation']][triple['subject']]]
#                 else:
#                     self.kg_rel_diff[triple['relation']][triple['subject']] = [None]
#
#             if type == 'change':
#                 self.update_new_kg_tuple(triple)
#                 old_text = ' '.join(
#                     [triple['subject'], triple['relation'], self.kg_rel[triple['relation']][triple['subject']]]).strip()
#                 if hash(old_text) in self.kg_set:
#                     self.kg_set.remove(hash(old_text))  # in case the novelty changed back!
#
#             self.kg_rel[triple['relation']][triple['subject']] = triple['object']  # update kg
#             self.kg_rel_diff[triple['relation']][triple['subject']].append(triple['object'])
#
#             return [triple['subject'], triple['relation'], self.kg_rel_diff[triple['relation']][triple['subject']]]
#
#     def kg_add(self, triple, level='sub', use_hash=False):
#         """
#         * Add game rule to KG by ***self.kg_add()***
#             * If we have this sub in this rel before, means kg changed, return True!
#             * If not, just add this to the kg graph. After adding the new game rule (no contradiction) to the big
#               **self.kg_rel**, we also call *self.update_new_kg_tuple()* to put this new rule in another dict,
#               **self.new_kg_tuple**.
#
#         :param triple (dict): triple is a dict with three keys: subject, relation and object
#         :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
#         :return: bool. True indicates the rule is changed, False means not changed or no existing rule, and add a rule
#         """
#         if level == 'sub':
#             if triple['subject'] in self.kg_sub.keys():
#                 if triple['relation'] in self.kg_sub[triple['subject']].keys():
#
#                     return (True, 'change') if self.kg_sub[triple['subject']][triple['relation']] != triple[
#                         'object'] else (False, None)
#                 else:
#                     self.kg_sub[triple['subject']][triple['relation']] = triple['object']
#
#             else:
#                 self.kg_sub[triple['subject']] = dict()
#                 self.kg_sub[triple['subject']][triple['relation']] = triple['object']
#             return (False, None)
#
#         else:  # level = 'rel'
#             if triple['relation'] in self.kg_rel.keys():
#                 if triple['subject'] in self.kg_rel[triple['relation']].keys():
#                     if use_hash:  # If we use hash, there is no need to check here, return True directly is ok
#                         if type(self.kg_rel[triple['relation']][triple['subject']]) == list:  # location
#                             # TODO: change the diff record way!!!
#                             return (True, 'change') if triple['object'] not in self.kg_rel[triple['relation']][
#                                 triple['subject']] else (False, None)
#                         return (True, 'change')
#                     else:
#                         return (True, 'change') if self.kg_rel[triple['relation']][triple['subject']] != triple[
#                             'object'] else (False, None)
#                     # True here means there is a change in the rule, False means stay the same.
#
#                 else:  # never see this rule, need update kg
#                     self.kg_rel[triple['relation']][triple['subject']] = triple['object']
#                     self.update_new_kg_tuple(triple)
#
#             else:  # never see this rel, update kg
#                 self.kg_rel[triple['relation']] = dict()
#                 self.kg_rel[triple['relation']][triple['subject']] = triple['object']
#                 self.update_new_kg_tuple(triple)
#
#             return (False, None) if self.update_num <= 4 * self.update_interval else (True, 'new')
#
#     def build_kg_file(self, file_name, level='sub', use_hash=False):
#         """
#         Give the logging file and add kg to the existing kg
#         1. put every sentence in *self.build_kg_text*
#         2.
#         :param file_name: logger info file path
#         :param level: 'sub' or 'rel'
#         :param use_hash: bool. Make the check of existing rule much faster
#         :return:
#         """
#
#         file = open(file_name, 'r')
#         for line in file:
#             kg_change = self.build_kg_text(line, level=level, use_hash=use_hash)  # difference -> tuple
#
#             # Check dice novelty
#             if self.text_dice_num == self.detection_num:
#                 self.dice.dice = self.dice.add_new_to_total_dice(self.dice.new_dice, self.dice.dice)
#             if self.text_dice_num > self.detection_num and self.text_dice_num % self.history_update_interval == 0:
#                 self.dice.run()
#                 self.text_dice_num = 1 + self.detection_num
#
#             if kg_change:
#                 print('kg_change', kg_change)
#                 print('self.kg_change:', self.kg_change)
#                 if kg_change[0] not in self.kg_change:
#                     self.kg_change += copy.deepcopy(kg_change)
#                     self.kg_change_bool = True
#
#         self.update_num += 1
#
#         # If there is any update or new relationships in kg, will update in the matrix
#
#         if self.update_num % self.update_interval == 0:
#
#             # solve difference of locations
#             if 'is located at' not in self.kg_rel.keys():
#                 self.kg_rel['is located at'] = dict()
#             else:
#                 diff = self.compare_loc_record(self.kg_rel['is located at'], self.location_record)
#                 print('diff ', diff)
#                 print('self.kg_change', self.kg_change)
#                 if diff and self.kg_change:
#                     for d in diff:
#                         exist_bool = False
#                         i = 0
#                         while i < len(self.kg_change):
#
#                             cha = self.kg_change[i]
#                             if cha[0] == d[0] and cha[1] == d[1]:
#                                 self.kg_change[i][-1][-1] = d[-1][-1]
#
#                                 if self.kg_change[i][-1][0] == self.kg_change[i][-1][-1]:
#                                     self.kg_change.pop(i)  # The novelty change back~ No need now
#                                 else:
#                                     i += 1
#
#                                 exist_bool = True
#
#                         if exist_bool == False:
#                             self.kg_change.append(d)
#
#             # for loc in self.location_record:
#             #     self.kg_rel['is located at'][loc] = self.location_record[loc]
#
#             self.kg_rel['is located at'] = self.location_record.copy()
#             self.location_record.clear()
#         if self.kg_change_bool or self.update_num == self.update_interval or self.update_num == self.update_interval in [
#             1, 2, 3, 4, 5]:
#             self.build_matrix_dict()
#             self.sparse_matrix = self.dict_to_matrix()
#             self.save_matrix()
#             self.kg_change_bool = False  # Reset new_kg_tuple
#             # Update history while only detect rule change after simulating 100 games
#         if self.dice.novelty != self.dice_novelty:
#             self.kg_change += self.dice.novelty
#             self.dice_novelty = self.dice.novelty[:]
#
#         if self.dice.novelty or self.kg_change:
#             return self.dice.type_record, self.kg_change
#         else:
#             return None
#
#     def build_kg_text(self, text, level='sub', use_hash=False):
#         """
#         Use a logging sentence to build or add or update kg
#         * _Check_
#                 * Check if the game rule **exists**, if yes, just ignore it.
#                 * Check if the game rule is about **location**.
#                 * Otherwise, add the game rule hash value to kg_set, so it will be easy to detect it in the future.
#         * _Add_
#             * Annotate the text with nlp server and get a tuple (sub, rel, obj).
#             * If location related, add location record to self.location_record, a dict.
#             * Add game rule to KG by ***self.kg_add()***
#                 * If we have this sub in this rel before, means kg changed, return True! If not, just add this to the kg graph.
#                 * After adding the new game rule to the big **self.kg_rel**, we also call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
#             * When adding the game rule, if we see contradiction, we will call ***self.kg_update()*** to update the knowledge graph and return difference. PS: we did not update the game rule when there is contradiction in *self.kg_add()*.
#                 * Update **self.kg_rel_diff** to record the difference.
#                 * Update knowledge graph, **self.kg_rel**
#                 * Call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
#                 * return the difference: [sub, rel, old-obj, new-obj]
#         * _Return_
#             * Return the difference with a list of lists, [[sub, rel, old-obj, new-obj], [sub, rel, old-obj, new-obj],...]
#
#         :param text: string. One sentence from logging info
#         :param level: string. 'sub' or 'rel'. level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
#         :return: list. Return the difference with a list of lists, [[sub, rel, old-obj, new-obj], [sub, rel, old-obj, new-obj],...]
#         """
#
#         diff = []
#         diff_set = set()
#         text = text.strip()
#         # Add history of dice => Novelty 1 - Dice
#         if 'Dice' in text and '[' in text:
#             self.text_dice_num += 1
#             dice_list = list(map(lambda x: int(x), text[text.index('[') + 1: text.index(']')].split(',')))  # i.e. [2,5]
#             self.dice.record_history_new_dice(dice_list)
#             return diff
#
#         # if 'Card' in text:
#         #     self.text_card_num += 1
#         #     card_name = text[text.index('>')+2:]
#         #     pack = text.split(' ')[0]
#         #     if pack != 'Chance':
#         #         pack = 'Community-Chest'
#         #     self.card.record_history_new_card(card_name, pack)
#         #     return diff
#
#         '''
#         Except dice, then record and check game rule => Novelty 1 - Card and Novelty 2
#         * _Check_
#             * Check if the game rule **exists**, if yes, just ignore it.
#             * Check if the game rule is about **location**.
#             * Otherwise, add the game rule hash value to kg_set, so it will be easy to detect it in the future.
#         '''
#
#         if use_hash:
#             triple_hash = hash(text)
#             if triple_hash in self.kg_set and 'locate' not in text:  # Record this rule previously, just skip
#                 return diff
#             elif triple_hash not in self.kg_set and 'locate' not in text:
#                 self.kg_set.add(triple_hash)  # Add to set for checking faster later on
#
#         # Annotate the text with nlp server and get a tuple (sub, rel, obj)
#         entity_relations = self.annotate(text, simple_format=True)
#
#         for er in entity_relations:  # er is a dict() containing sub, rel and obj
#             if 'locate' in text:  # about location
#                 self.add_loc_history(er)  # after some rounds, we will compare the loc history in build_file
#                 return diff  # return no diff
#             else:  # other rules
#                 kg_change_once, type_update = self.kg_add(er, level=level,
#                                                           use_hash=use_hash)  # kg_change_once is a bool, True means rule change
#             if kg_change_once:  # Bool
#                 diff_once = self.kg_update(er, level=level, type=type_update)  # find difference
#                 if diff_once:
#                     diff.append(diff_once)
#         return diff
#
#     def add_loc_history(self, triple):
#         """
#         Add location info to subject => dict()
#         :param triple: dict()
#         :return: None
#         """
#         if triple['subject'] in self.location_record:
#             if triple['object'] not in self.location_record[triple['subject']]:
#                 self.location_record[triple['subject']].append(triple['object'])
#         else:
#             self.location_record[triple['subject']] = [triple['object']]
#
#     def compare_loc_record(self, total, new):
#         """
#         Compare two location dict()'s difference
#         :param total: old big dict()
#         :param new: new small dict()
#         :return: diff : list().
#         """
#         diff = []
#         for space in total.keys():
#             if space in new:
#                 if sorted(total[space]) != sorted(new[space]):
#                     diff.append([space, 'is located at', [sorted(total[space]), sorted(new[space])]])
#         for space in new.keys():
#             if space not in total.keys():
#                 diff.append([space, 'is located at', [[], sorted(new[space])]])
#         return diff
#
#     def generate_graphviz_graph_(self, text: str = '', png_filename: str = './out/graph.png', level: str = 'acc',
#                                  kg_level='rel'):
#         """
#         Plot the knowledge graph with exsiting kg
#        :param (str | unicode) text: raw text for the CoreNLPServer to parse
#        :param (list | string) png_filename: list of annotators to use
#        :param (str) level: control we plot the whole image all the local knowledge graph
#        """
#         entity_relations = self.annotate(text, simple_format=True)
#         """digraph G {
#         # a -> b [ label="a to b" ];
#         # b -> c [ label="another label"];
#         }"""
#         if level == 'single':
#             graph = list()
#             graph.append('digraph {')
#             for er in entity_relations:
#                 kg_change, type_update = self.kg_add(er)
#                 graph.append('"{}" -> "{}" [ label="{}" ];'.format(er['subject'], er['object'], er['relation']))
#             graph.append('}')
#         else:
#             graph = list()
#             graph.append('digraph {')
#             if kg_level == 'rel':
#                 for rel in self.kg_rel.keys():
#                     for sub in self.kg_rel[rel]:
#                         graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_rel[rel][sub], rel))
#             else:
#                 for sub in self.kg_sub.keys():
#                     for rel in self.kg_sub[sub]:
#                         graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_sub[sub][rel], rel))
#             graph.append('}')
#
#         output_dir = os.path.join('.', os.path.dirname(png_filename))
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         out_dot = os.path.join(tempfile.gettempdir(), 'graph.dot')
#         with open(out_dot, 'w') as output_file:
#             output_file.writelines(graph)
#
#         command = 'dot -Tpng {} -o {}'.format(out_dot, png_filename)
#         dot_process = Popen(command, stdout=stderr, shell=True)
#         dot_process.wait()
#         assert not dot_process.returncode, 'ERROR: Call to dot exited with a non-zero code status.'
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass
#
#     def __del__(self):
#         self.client.stop()
#         del os.environ['CORENLP_HOME']
#
#         # def save_json(self, save_dict):
#         #     '''
#         #     Save kg dict to json file
#         #     :param level:
#         #     :return: None
#         #     '''
#         #     import json
#         #     if save_level == 'total':
#         #         if level == 'sub':
#         #             with open(self.jsonfile, 'w') as f:
#         #                 json.dump(self.kg_sub, f)
#         #         else:
#         #             with open(self.jsonfile, 'w') as f:
#         #                 json.dump(self.kg_rel, f)
#
#         # def read_json(self, level='sub'):
#         #     '''
#         #     Read kg dict file from json file
#         #     :param level:
#         #     :return: None
#         #     '''
#         #     import json
#         #     with open(self.jsonfile, 'r') as f:
#         #         if level == 'sub':
#         #             self.kg_sub = json.load(f)
#         #         else:
#         #             self.kg_rel = json.load(f)
#
#         # only kg_rel needs sparse matrix
#
#     def build_matrix_dict(self):
#         '''
#         build a dict for building sparse matrix
#         names - price - price_1_house - rents....
#         '''
#         for i, rel in enumerate(self.relations_matrix):  # ['is priced at', ...]
#             if rel in self.kg_rel.keys():
#                 for sub in self.kg_rel[rel].keys():
#                     if sub in self.sparse_matrix_dict['nodes_number']:
#                         sub_id = self.sparse_matrix_dict['nodes_number'][sub]
#                         if type(sub_id) == int:
#                             sub_id = [sub_id]
#                         for number_sub in sub_id:
#                             # Adjust for more rel/edges: TODO
#                             # if i == 0:
#                             number_obj = self.sparse_matrix_dict['nodes_number'][
#                                 self.relations_node[i] + str(self.hash_money(int(self.kg_rel[rel][sub]), i))]
#                             # else:
#                             #     if len(self.kg_rel[rel][sub]) > 1:
#                             #         number_obj = []
#                             #         for location in self.kg_rel[rel][sub]:
#                             #             if self.relations_node[i] + str(location) in self.sparse_matrix_dict['nodes_number']:
#                             #                 number_obj.append(int(self.sparse_matrix_dict['nodes_number'][self.relations_node[i] + str(location)]))
#                             #     else:
#                             #         number_obj = int(self.sparse_matrix_dict['nodes_number'][
#                             #             self.relations_node[i] + str(self.kg_rel[rel][sub][0])])
#
#                             self.sparse_matrix_dict['out'][rel][number_sub] = number_obj if type(
#                                 number_obj) == list else [number_obj]
#
#                             # if type(number_obj) == list:
#                             #     for location in number_obj:
#                             #         self.sparse_matrix_dict['in'][rel][location] = [int(number_sub)]
#                             # else:
#                             #     self.sparse_matrix_dict['in'][rel][number_obj] = [int(number_sub)]
#         print(self.sparse_matrix_dict['out'])
#
#     def dict_to_matrix(self):
#         self.sparse_matrix = []
#         for i in range(self.node_number):  # node_id
#             self.sparse_matrix.append([])
#             for type in ['out']:
#                 for rel in self.relations_matrix:
#
#                     node_1 = self.sparse_matrix_dict[type][rel][i] if i in self.sparse_matrix_dict[type][rel] else None
#                     matrix_1 = [0 for j in range(self.node_number)]
#                     if node_1:
#
#                         if len(node_1) == 1:
#                             matrix_1[node_1[0]] = 1 if node_1[0] > 0 else 0
#                         else:
#                             for loc in node_1:
#                                 matrix_1[loc] = 1
#
#                     self.sparse_matrix[-1] += matrix_1
#
#             # add color relationship like a and b share the same color so they will have a relationship
#             name_i = self.sparse_matrix_dict['number_nodes'][i]
#             matrix_1 = [0 for j in range(self.node_number)]
#             if name_i in self.kg_rel['is colored as']:
#                 color_i = self.kg_rel['is colored as'][name_i]
#                 for sub, obj in self.kg_rel['is colored as'].items():
#                     if obj == color_i and sub != name_i:
#                         sub_number = self.sparse_matrix_dict['nodes_number'][sub]
#                         matrix_1[sub_number] = 1
#             self.sparse_matrix[-1] += matrix_1
#
#         self.sparse_matrix = np.array(self.sparse_matrix)  # save as np array
#         # for rel in self.action_name:
#         #     self.sparse_matrix.append(csr_matrix((self.sparse_matrix_dict[rel]['data'], (self.sparse_matrix_dict[rel]['row'], self.sparse_matrix_dict[rel]['col'])), shape=(self.entity_num, self.entity_num)))
#         return self.sparse_matrix
#
#     def update_new_kg_tuple(self, triple):
#         '''
#         Update self.new_kg_tuple when there is new rule in kg
#         :param triple: new kg rule tuple
#         '''
#         if triple['relation'] in self.new_kg_tuple.keys():
#             pass
#         else:
#             self.new_kg_tuple[triple['relation']] = dict()
#
#         self.new_kg_tuple[triple['relation']][triple['subject']] = triple['object']
#
#     def save_matrix(self):
#         '''
#         Save sparse matrix of kg
#         :return:
#         '''
#         print('self.matrix_file_path', self.matrix_file_path)
#         np.save(self.matrix_file_path, self.sparse_matrix)
#         # node_id = dict()
#         # for node in self.sparse_matrix_dict['nodes_number']:
#         #     if type(self.sparse_matrix_dict['nodes_number'][node]) == list:
#         #         for i in range(len(self.sparse_matrix_dict['nodes_number'][node])):
#         #             node_id[node + '_' + str(i)] = self.sparse_matrix_dict['nodes_number'][node][i]
#         #     else:
#         #         node_id[node] = self.sparse_matrix_dict['nodes_number'][node]
#         #
#         #
#         # self.save_json(node_id, self.entity_file_path)
#         self.save_json(self.sparse_matrix_dict['number_nodes'], self.entity_file_path)
#
#     def save_vector(self):
#         np.save(self.vector_file, self.kg_vector)
#
#     def build_vector(self):
#         '''
#         Build the representation vector using knowledge graph
#         '''
#         num = 0
#         for rel in self.relations_matrix:
#             if rel in self.new_kg_tuple.keys():
#                 for sub in self.new_kg_tuple[rel].keys():
#                     index_sub = int(self.board_name.index(sub))
#                     obj = self.new_kg_tuple[rel][sub]
#                     self.kg_vector[num][index_sub] = int(obj)
#             num += 1
#
#     # def record_history(self, name, history_dict):
#     #     if name in history_dict.keys():
#     #         history_dict[name]
#
# class Novelty_Detection_Dice(History_Record):
#     def __init__(self, config_file=None):
#         # Novelty Detection
#         # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
#         self.upper_path = '/media/becky/GNOME-p3'
#         # if config_file == None:
#         #     config_file = self.upper_path + '/monopoly_simulator_background/config.ini'
#
#         config_data = ConfigParser()
#         config_data.read(config_file)
#         self.novelty_params = self.params_read(config_data, keys='novelty')
#         self.new_dice = dict()
#         self.dice = dict()
#         self.percentage_var = self.novelty_params['percentage_var']
#         self.num_dice = 0
#         self.state_dice = []
#         self.type_dice = []
#         self.novelty = []
#         self.type_record = [dict(), dict()]
#
#         # For safety
#         self.new_dice_temp = dict()
#         self.temp_bool = False
#
#     def run(self):
#         '''
#          = main function in this class
#         :return: A list with tuples, including the dice novelty
#         '''
#
#         if self.temp_bool:
#             novelty_temp = self.compare_dice_novelty(self.new_dice_temp, self.new_dice)
#             novelty_temp_total = self.compare_dice_novelty(self.new_dice, self.dice)
#             if novelty_temp_total == []:
#                 self.dice = self.add_new_to_total_dice(self.new_dice, self.dice)
#                 self.new_dice_temp.clear()
#                 self.new_dice.clear()
#                 self.temp_bool = False
#                 return []
#             else:
#                 if novelty_temp:
#                     self.temp_bool = True
#                     self.new_dice_temp = self.new_dice.copy()
#                     self.new_dice.clear()
#                     return []
#                 else:
#                     self.novelty.append(novelty_temp_total)
#                     # print('temp', self.new_dice_temp, self.new_dice)
#                     # self.type_record[1] = {'num': num_dice_new, 'state': state_dice_new, 'type': type_dice_new,
#                     #                        'percentage': percentage_new}
#                     # self.type_record[0] = {'num': num_dice, 'state': state_dice, 'type': type_dice, 'percentage': percentage}
#                     self.dice.clear()
#                     self.dice = self.add_new_to_total_dice(self.new_dice, self.dice)
#                     self.dice = self.add_new_to_total_dice(self.new_dice_temp, self.dice)
#                     self.temp_bool = False
#                     self.new_dice.clear()
#                     self.new_dice_temp.clear()
#                     return novelty_temp_total
#
#         else:
#             novelty = self.compare_dice_novelty(self.new_dice, self.dice)
#
#             if self.temp_bool:
#                 self.new_dice.clear()
#                 return []
#             else:
#                 self.dice = self.add_new_to_total_dice(self.new_dice, self.dice)
#
#             return novelty
#
#     def record_history_new_dice(self, dice_list):
#         '''
#         Record the history of dice to new_dice dict
#         :param dice_list (list):  a list indicating the dice from logging i.e. [2,3]
#         :return: None
#         '''
#         for i, num in enumerate(dice_list):
#             if i in self.new_dice.keys():
#                 if num in self.new_dice[i].keys():
#                     self.new_dice[i][num] += 1
#                 else:
#                     self.new_dice[i][num] = 1
#             else:
#                 self.new_dice[i] = dict()
#                 self.new_dice[i][num] = 1
#
#     def add_new_to_total_dice(self, new, total):
#         for key in total.keys():
#             if key in new.keys():
#                 total[key] = dict(Counter(total[key]) + Counter(new[key]))
#         for key in new.keys():
#             if key not in total.keys():
#                 total[key] = new[key]
#         new.clear()
#         return total
#
#     def dice_evaluate(self, evaluated_dice_dict):
#         '''
#         Evaluate dice type, state, number
#         :param evaluated_dice_dict (dict): put a dice history in dict
#         :return: num_dice: # of dice used
#                 state_dice: state of each dice
#                 type_dice: dice are biased or uniform
#         '''
#         num_dice = len(evaluated_dice_dict.keys())  # int : 2
#         state_dice = []  # [[1,2,3],[1,2]]
#         type_dice = []
#         percentages = []
#         for key in evaluated_dice_dict.keys():
#             state = list(map(lambda x: x[0], sorted(list(evaluated_dice_dict[key].items()), key=lambda x: x[0])))
#             state_dice.append(state)
#             nums = list(map(lambda x: x[1], sorted(list(evaluated_dice_dict[key].items()), key=lambda x: x[0])))
#             percentage = [num / sum(nums) for num in nums]
#
#             # Use KS-test to evaluate dice type:
#             test_list = []
#             for i, state_number in enumerate(state):
#                 test_list += [state_number for j in range(nums[i])]
#
#             num_ks = 0
#             p_value_com = 0
#             while num_ks < 5:
#                 num_ks += 1
#                 test_distri = []
#                 for i, state_number in enumerate(state):
#                     test_distri += [state_number for j in range(int(sum(nums) / len(state)))]
#
#                 p_value = stats.ks_2samp(np.array(test_list), np.array(test_distri)).pvalue
#                 p_value_com = max(p_value_com, p_value)
#                 if p_value_com > self.percentage_var:
#                     break
#
#             if p_value_com <= self.percentage_var:
#                 type_dice.append('Bias')
#             else:
#                 type_dice.append('Uniform')
#                 percentage = [1 / len(nums)]
#
#             percentages.append(percentage)
#
#         return num_dice, state_dice, type_dice, percentages
#
#     def compare_dice_novelty(self, new_dice, dice):
#         '''
#         Dice Novelty Detection Type
#         1. state
#         2. type
#         :return: bool. True means detecting novelty
#         '''
#         # print('self.new_dice',self.new_dice)
#         # print('self.dice', self.dice)
#         dice_novelty_list = []
#         # Detect new state of dice. i.e. [1,2,3,4] => [1,2,3,4,5], we have a 5 now
#         num_dice_new, state_dice_new, type_dice_new, percentage_new = self.dice_evaluate(new_dice)
#         num_dice, state_dice, type_dice, percentage = self.dice_evaluate(dice)
#
#         if num_dice_new != num_dice:
#             dice_novelty_list.append(('Num', num_dice_new, num_dice))
#         if state_dice_new != state_dice:
#             dice_novelty_list.append(('State', state_dice_new, state_dice))
#         if type_dice_new != type_dice:
#             dice_novelty_list.append(('Type', type_dice_new, percentage_new, type_dice, percentage))
#
#         if dice_novelty_list:
#             # When U detect sth. new, do not tell the agent immediately
#             if self.temp_bool == False:
#                 self.new_dice_temp = new_dice.copy()  # Record this and compare later
#                 self.temp_bool = True
#                 return []
#             else:
#                 # self.novelty.append(dice_novelty_list)
#                 #
#                 # if dice == self.dice:
#                 #     self.type_record[1] = {'num': num_dice_new, 'state': state_dice_new, 'type': type_dice_new,
#                 #                            'percentage': percentage_new}
#                 #     self.type_record[0] = {'num': num_dice, 'state': state_dice, 'type': type_dice, 'percentage': percentage}
#                 # self.temp_bool = False
#
#                 # print('dice_novelty_list',dice_novelty_list)
#                 return dice_novelty_list
#
# class Novelty_Detection_Card(History_Record):
#     def __init__(self, config_file=None):
#         # Novelty Detection
#         # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
#         self.upper_path = '/media/becky/GNOME-p3'
#         # if config_file == None:
#         #     config_file = self.upper_path + '/monopoly_simulator_background/config.ini'
#
#         config_data = ConfigParser()
#         config_data.read(config_file)
#         self.novelty_params = self.params_read(config_data, keys='novelty')
#         self.card = dict()
#         self.new_card = dict()
#
#     def record_history_new_card(self, card_name, pack):
#         if pack not in self.new_card.keys():
#             self.new_card[pack] = dict()
#         if card_name not in self.new_card[pack].keys():
#             self.new_card[pack][card_name] = 1
#         else:
#                 self.new_card[pack][card_name] += 1

class KG_OpenIE_eva(History_Record):
    def __init__(self, gameboard,
                 matrix_file_name,
                 entity_file_name,
                 core_nlp_version: str = '2018-10-05',
                 config_file=None):

        self.upper_path = upper_path
        # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')

        # nlp server env
        self.remote_url = 'https://nlp.stanford.edu/software/stanford-corenlp-full-{}.zip'.format(core_nlp_version)
        self.install_dir = Path('~/.stanfordnlp_resources/').expanduser()
        self.install_dir.mkdir(exist_ok=True)
        if not (self.install_dir / Path('stanford-corenlp-full-{}'.format(core_nlp_version))).exists():
            print('Downloading from %s.' % self.remote_url)
            output_filename = wget.download(self.remote_url, out=str(self.install_dir))
            print('\nExtracting to %s.' % self.install_dir)
            zf = ZipFile(output_filename)
            zf.extractall(path=self.install_dir)
            zf.close()
        os.environ['CORENLP_HOME'] = str(self.install_dir / 'stanford-corenlp-full-2018-10-05')
        self.client = CoreNLPClient(annotators=['openie'], memory='8G')

        config_data = ConfigParser()
        config_data.read(config_file)
        self.params = self.params_read(config_data, keys='kg')
        self.jsonfile = self.upper_path + self.params['jsonfile']
        self.relations = ['price', 'rent', 'located', 'colored', 'classified', 'away', 'type', 'cost', 'direct', 'mortgaged', 'increment']
        self.relations_matrix = ['is priced at', 'is price-1-house at', 'is rented-0-house at', 'is rented-1-house at', \
                                 'is rented-2-house at', 'is rented-3-house at', 'is rented-4-house at', 'is rented-1-hotel at'] #, 'is colored as', 'is classified as']
        self.relations_node = ['Price_', 'Price_1_house_', 'Rent_0_house_', 'Rent_1_house_', 'Rent_2_house_', 'Rent_3_house_',\
                               'Rent_4_house_', 'Rent_1_hotel_']
        self.under_list = []
        self.upper_list = []
        #####################
        self.price_list_under = [0, 100, 150, 200, 300, 400, 1000]
        self.under_list.append(self.price_list_under)
        self.price_list_upper = [99, 149, 199, 299, 399, 999, sys.maxsize]
        self.upper_list.append(self.price_list_upper)
        #####################
        self.price_1_house_list_under = [0, 50, 100, 150, 200]
        self.under_list.append(self.price_1_house_list_under )
        self.price_1_house_list_upper = [49, 99, 149, 199, sys.maxsize]
        self.upper_list.append(self.price_1_house_list_upper)
        #####################
        self.rent_0_house_list_under = [0, 5, 10, 15, 20, 30, 50]
        self.under_list.append(self.rent_0_house_list_under)
        self.rent_0_house_list_upper = [4, 9, 14, 19, 29, 49, sys.maxsize]
        self.upper_list.append(self.rent_0_house_list_upper)
        #####################
        self.rent_1_house_list_under = [0, 50, 100, 150, 200]
        self.under_list.append(self.rent_1_house_list_under)
        self.rent_1_house_list_upper = [49, 99, 149, 199, sys.maxsize]
        self.upper_list.append(self.rent_1_house_list_upper)
        #####################
        self.rent_2_house_list_under = [0, 50, 100, 200, 300, 400, 500, 600]
        self.under_list.append(self.rent_2_house_list_under)
        self.rent_2_house_list_upper = [49, 99, 199, 299, 399, 499, 599, sys.maxsize]
        self.upper_list.append(self.rent_2_house_list_upper)
        #####################
        self.rent_3_house_list_under = [0, 100, 300, 500, 800, 1000, 1200, 1400]
        self.under_list.append(self.rent_3_house_list_under)
        self.rent_3_house_list_upper = [99, 299, 499, 799, 999, 1199, 1399, sys.maxsize]
        self.upper_list.append(self.rent_3_house_list_upper)
        #####################
        self.rent_4_house_list_under = [0, 200, 500, 800, 1200, 1700]
        self.under_list.append(self.rent_4_house_list_under)
        self.rent_4_house_list_upper = [199, 499, 799, 1199, 1699, sys.maxsize]
        self.upper_list.append(self.rent_4_house_list_upper)
        #####################
        self.rent_1_hotel_list_under = [0, 300, 600, 1000, 1500, 2000]
        self.under_list.append(self.rent_1_hotel_list_under)
        self.rent_1_hotel_list_upper = [299, 599, 999, 1499, 1999, sys.maxsize]
        self.upper_list.append(self.rent_1_hotel_list_upper)
        #####################



        self.kg_rel = dict()  # the total kg rule for "rel" KG
        self.kg_sub = dict()  # the total kg rule for "sub" KG
        self.kg_set = set()   # the set recording the kg rule for searching if the rule exists quickly
        self.kg_rel_diff = dict()  # the dict() to record the rule change
        self.kg_sub_diff = dict()
        self.kg_change_bool = False

        self.kg_introduced = False
        self.new_kg_tuple = dict()
        self.update_num = 1
        self.update_interval = self.params['update_interval']
        self.detection_num = self.params['detection_num']
        self.kg_change = []
        self.history_update_interval = self.params['history_update_interval']
        self.location_record = dict()

        # For kg to matrix
        self.matrix_params = self.params_read(config_data, keys='matrix')
        self.entity_num = self.matrix_params['entity_num']
        self.action_num = self.matrix_params['action_num']
        self.sparse_matrix = []
        self.action_name = ['is ' + str(i) +'-step away from' for i in range(1,41)]
        self.board_name = []

        self.set_gameboard(gameboard) #get info about the gameboard each round
        # self.board_name = ['Go','Mediterranean-Avenue', 'Community Chest-One',
        #         'Baltic-Avenue', 'Income Tax', 'Reading Railroad', 'Oriental-Avenue',
        #         'Chance-One', 'Vermont-Avenue', 'Connecticut-Avenue', 'In-Jail/Just-Visiting',
        #         'St. Charles Place', 'Electric Company', 'States-Avenue', 'Virginia-Avenue',
        #         'Pennsylvania Railroad', 'St. James Place', 'Community Chest-Two', 'Tennessee-Avenue',
        #         'New-York-Avenue', 'Free Parking', 'Kentucky-Avenue', 'Chance-Two', 'Indiana-Avenue',
        #         'Illinois-Avenue', 'B&O Railroad', 'Atlantic-Avenue', 'Ventnor-Avenue',
        #         'Water Works', 'Marvin Gardens', 'Go-to-Jail', 'Pacific-Avenue', 'North-Carolina-Avenue',
        #         'Community Chest-Three', 'Pennsylvania-Avenue', 'Short Line', 'Chance-Three', 'Park Place',
        #                                 'Luxury Tax', 'Boardwalk']
        self.node_number = 0
        self.sparse_matrix_dict = self.build_empty_matrix_dict()
        self.matrix_folder = self.upper_path + self.matrix_params['matrix_folder']
        self.matrix_file_path = matrix_file_name
        self.entity_file_path = entity_file_name
        self.kg_vector = np.zeros([len(self.relations_matrix), len(self.board_name)])
        self.vector_file = self.upper_path + self.matrix_params['vector_file']

        # Dice Novelty
        self.dice = Novelty_Detection_Dice(config_file)
        self.text_dice_num = 0
        self.dice_novelty = []
        self.card_board_novelty = []
        # self.text_card_num = 0
        # self.card = Novelty_Detection_Card(config_file)
        self.card_board = Novelty_Detection_Card_Board()
        self.card_board.ini_cards()

    def set_gameboard(self, gameboard):
        self.gameboard = gameboard
        self.board_name = gameboard['location_sequence']
        for i, name in enumerate(self.board_name):
            self.board_name[i] = '-'.join(name.split(' '))

    # def build_empty_matrix_dict(self):
    #     sparse_matrix_dict = dict()
    #     for rel in self.action_name:
    #         sparse_matrix_dict[rel] = dict()
    #         sparse_matrix_dict[rel]['row'] = []
    #         sparse_matrix_dict[rel]['col'] = []
    #         sparse_matrix_dict[rel]['data'] = []
    #     return sparse_matrix_dict

    def hash_money(self, money, index):
        under_list = self.under_list[index]
        upper_list = self.upper_list[index]
        for i, under_limit in enumerate(under_list):
            if money >= under_limit and money <= upper_list[i]:
                return under_limit

        return -1

    def build_empty_matrix_dict(self):
        """
        Build a empty dict for storing the matrix
        :return: a dict
        """
        sparse_matrix_dict = dict()
        sparse_matrix_dict['number_nodes'] = dict()
        sparse_matrix_dict['nodes_number'] = dict()
        sparse_matrix_dict['out'] = dict()
        sparse_matrix_dict['in'] = dict()
        sparse_matrix_dict['number_rel'] = dict()
        sparse_matrix_dict['rel_number'] = dict()
        sparse_matrix_dict['all_rel'] = dict()

        # Define the relation/edge number
        for i,rel in enumerate(self.relations_matrix):
            sparse_matrix_dict['number_rel'][i] = rel
            sparse_matrix_dict['rel_number'][rel] = i

        # Define the number of nodes
        # name of location

        for i, node in enumerate(self.board_name):
            sparse_matrix_dict['number_nodes'][i] = node
            if node in sparse_matrix_dict['nodes_number']:
                id_value = sparse_matrix_dict['nodes_number'][node]
                if type(id_value) == list:
                    id_value.append(i)
                else:
                    id_value = [id_value] + [i]
                sparse_matrix_dict['nodes_number'][node] = id_value
            else:
                sparse_matrix_dict['nodes_number'][node] = i
        index_now = i + 1

        # [7]Price: 0-99; 100-149;150-199;200-299;300-399;400-999; 1000+
        for i, price in enumerate(self.price_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Price_' + str(price)
            sparse_matrix_dict['nodes_number']['Price_' + str(price)] = i + index_now
        index_now += i + 1

        #[5]Price: 1 - house: 0-49; 50-99; 100-149; 150-199; 200+
        for i, price in enumerate(self.price_1_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Price_1_house_' + str(price)
            sparse_matrix_dict['nodes_number']['Price_1_house_' + str(price)] = i + index_now
        index_now += i + 1
        # #location
        # for k, loc in enumerate(range(40)):
        #     sparse_matrix_dict['number_nodes'][i + j + k + 2] = 'Location_' + str(k)
        #     sparse_matrix_dict['nodes_number']['Location_' + str(k)] = i + j + k + 2

        # [7] Rent_0_house: 0-4, 5-9, 10-14, 15-19, 20-29, 30-49, 50+
        for i, rent in enumerate(self.rent_0_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_0_house_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_0_house_' + str(rent)] = i + index_now
        index_now += i + 1  # update the index for next node

        # [5] Rent_1_house: 0-49, 50-99, 100-149, 150-199, 200+
        for i, rent in enumerate(self.rent_1_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_1_house_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_1_house_' + str(rent)] = i + index_now
        index_now += i + 1  # update the index for next node

        # [8] Rent_2_house: 0-49, 50-99, 100-199, 200-299, 300-399,400-499,500-599,600+
        for i, rent in enumerate(self.rent_2_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_2_house_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_2_house_' + str(rent)] = i + index_now
        index_now += i + 1  # update the index for next node

        # [8] Rent_3_house: 0-99, 100-299, 300-499, 500-799, 800-999,1000-1199,1200-1399,1400+
        for i, rent in enumerate(self.rent_3_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_3_house_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_3_house_' + str(rent)] = i + index_now
        index_now += i + 1  # update the index for next node

        # [6] Rent_4_house: 0-199, 200-499, 500-799, 800-1199,1200-1699,1700+
        for i, rent in enumerate(self.rent_4_house_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_4_house_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_4_house_' + str(rent)] = i + index_now
        index_now += i + 1

        # [6] Rent_1_hotel: 0-299, 300-599, 600-999, 1000-1499,1500-1999,2000+
        for i, rent in enumerate(self.rent_1_hotel_list_under):
            sparse_matrix_dict['number_nodes'][i + index_now] = 'Rent_1_hotel_' + str(rent)
            sparse_matrix_dict['nodes_number']['Rent_1_hotel_' + str(rent)] = i + index_now

        # # Define 'in' column names
        # for rel in self.relations_matrix:
        #     sparse_matrix_dict['in'][rel] = dict()
        #     for node in range(self.node_number):
        #         sparse_matrix_dict['in'][rel][node] = None

        self.node_number = len(list(sparse_matrix_dict['number_nodes'].keys()))

        # Define 'out' column names
        for rel in self.relations_matrix:
            sparse_matrix_dict['out'][rel] = dict()
            for node in range(self.node_number):
                sparse_matrix_dict['out'][rel][node] = None

        # for rel in self.action_name:
        #     sparse_matrix_dict[rel] = dict()
        #     sparse_matrix_dict[rel]['row'] = []
        #     sparse_matrix_dict[rel]['col'] = []
        #     sparse_matrix_dict[rel]['data'] = []
        return sparse_matrix_dict

    def annotate(self, text: str, properties_key: str = None, properties: dict = None, simple_format: bool = True):
        """
        Annotate text to triples: sub, rel, obj
        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (str) properties_key: key into properties cache for the client
        :param (dict) properties: additional request properties (written on top of defaults)
        :param (bool) simple_format: whether to return the full format of CoreNLP or a simple dict.
        :return: Depending on simple_format: full or simpler format of triples <subject, relation, object>.
        """
        # https://stanfordnlp.github.io/CoreNLP/openie.html

        text = text.replace('_', '-') #Some words in logger info containiis not detected in openie, hence we replace '_' to '-'

        core_nlp_output = self.client.annotate(text=text, annotators=['openie'], output_format='json',
                                               properties_key=properties_key, properties=properties)
        if simple_format:
            triples = []
            for sentence in core_nlp_output['sentences']:
                for triple in sentence['openie']:

                    for rel in self.relations:
                        length_sub = len(triple['subject'].split(' '))
                        length_obj = len(triple['object'].split(' '))
                        if rel in triple['relation'] and length_sub == 1 and length_obj == 1 and \
                                triple['subject'] in self.board_name:

                            #map B&0-Railroad
                            if 'B-' in triple['subject']:
                                triple['subject'] = 'B&O-Railroad'
                            if 'B-' in triple['object']:
                                triple['object'] = 'B&O-Railroad'
                            if triple['subject'] == 'GO':
                                triple['subject'] = 'Go'
                            triples.append({
                                'subject': triple['subject'],
                                'relation': triple['relation'],
                                'object': triple['object']
                            })

                    # triples.append({
                    #     'subject': triple['subject'],
                    #     'relation': triple['relation'],
                    #     'object': triple['object']
                    # })
            return triples
        else:
            return core_nlp_output

    def kg_update(self, triple, level='sub', type=None):
        """
        After detecting rule change, update kg and also return the diff of kg
        * When adding the game rule, if we see contradiction, we will call ***self.kg_update()*** to update the knowledge
          graph and return difference. PS: we did not update the game rule when there is contradiction in *self.kg_add()*.
                    * Update **self.kg_rel_diff** to record the difference.
                    * Update knowledge graph, **self.kg_rel**
                    * Call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
                    * return the difference: [sub, rel, old-obj, new-obj]

        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: A tuple (sub, rel, diff)
        """
        if level == 'sub':
            if triple['subject'] not in self.kg_sub_diff.keys():
                self.kg_sub_diff[triple['subject']] = dict()
                self.kg_sub_diff[triple['subject']][triple['relation']] = [self.kg_sub[triple['subject']][triple['relation']]]
            self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            self.kg_sub_diff[triple['subject']][triple['relation']].append(triple['object'])
            return [triple['subject'],triple['relation'],self.kg_sub_diff[triple['subject']][triple['relation']]]

        else:
            if triple['relation'] not in self.kg_rel_diff.keys():
                self.kg_rel_diff[triple['relation']] = dict()

            if triple['subject'] not in self.kg_rel_diff[triple['relation']].keys():
                if type == 'change':
                    self.kg_rel_diff[triple['relation']][triple['subject']] = [self.kg_rel[triple['relation']][triple['subject']]]
                else:
                    self.kg_rel_diff[triple['relation']][triple['subject']] = [None]

            if type == 'change':
                self.update_new_kg_tuple(triple)
                old_text = ' '.join([triple['subject'], triple['relation'], self.kg_rel[triple['relation']][triple['subject']]]).strip()
                if hash(old_text) in self.kg_set:
                    self.kg_set.remove(hash(old_text))  # in case the novelty changed back!

            self.kg_rel[triple['relation']][triple['subject']] = triple['object']  # update kg
            self.kg_rel_diff[triple['relation']][triple['subject']].append(triple['object'])


            return [triple['subject'], triple['relation'], self.kg_rel_diff[triple['relation']][triple['subject']]]


    def kg_add(self, triple, level='sub', use_hash=False):
        """
        * Add game rule to KG by ***self.kg_add()***
            * If we have this sub in this rel before, means kg changed, return True!
            * If not, just add this to the kg graph. After adding the new game rule (no contradiction) to the big
              **self.kg_rel**, we also call *self.update_new_kg_tuple()* to put this new rule in another dict,
              **self.new_kg_tuple**.

        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed, False means not changed or no existing rule, and add a rule
        """
        if level == 'sub':
            if triple['subject'] in self.kg_sub.keys():
                if triple['relation'] in self.kg_sub[triple['subject']].keys():

                    return (True, 'change') if self.kg_sub[triple['subject']][triple['relation']] != triple['object'] else (False, None)
                else:
                    self.kg_sub[triple['subject']][triple['relation']] = triple['object']

            else:
                self.kg_sub[triple['subject']] = dict()
                self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            return (False, None)

        else:  # level = 'rel'
            if triple['relation'] in self.kg_rel.keys():
                if triple['subject'] in self.kg_rel[triple['relation']].keys():
                    if use_hash:  # If we use hash, there is no need to check here, return True directly is ok
                        if type(self.kg_rel[triple['relation']][triple['subject']]) == list:  # location
                            # TODO: change the diff record way!!!
                            return (True, 'change') if triple['object'] not in self.kg_rel[triple['relation']][triple['subject']] else (False, None)
                        return (True, 'change')
                    else:
                        return (True, 'change') if self.kg_rel[triple['relation']][triple['subject']] != triple['object'] else (False, None)
                    # True here means there is a change in the rule, False means stay the same.

                else:  # never see this rule, need update kg
                    self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                    self.update_new_kg_tuple(triple)

            else:  # never see this rel, update kg
                self.kg_rel[triple['relation']] = dict()
                self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                self.update_new_kg_tuple(triple)

            return (False, None) if self.update_num <= 4 * self.update_interval else (True, 'new')

    def build_kg_file(self, file_name, level='sub', use_hash=False):
        """
        Give the logging file and add kg to the existing kg
        1. put every sentence in *self.build_kg_text*
        2.
        :param file_name: logger info file path
        :param level: 'sub' or 'rel'
        :param use_hash: bool. Make the check of existing rule much faster
        :return:
        """

        file = open(file_name, 'r')
        for line in file:
            kg_change = self.build_kg_text(line, level=level, use_hash=use_hash)  # difference -> tuple

            # Check dice novelty
            if self.text_dice_num == self.detection_num:
                self.dice.dice = self.dice.add_new_to_total_dice(self.dice.new_dice, self.dice.dice)
            if self.text_dice_num > self.detection_num and self.text_dice_num % self.history_update_interval == 0:
                self.dice.run()
                self.text_dice_num = 1 + self.detection_num

            if kg_change:
                # print('kg_change', kg_change)
                # print('self.kg_change:', self.kg_change)
                if kg_change[0] not in self.kg_change:
                    self.kg_change += copy.deepcopy(kg_change)
                    self.kg_change_bool = True

        self.update_num += 1

        # If there is any update or new relationships in kg, will update in the matrix

        if self.update_num % self.update_interval == 0:

            #solve difference of locations
            if 'is located at' not in self.kg_rel.keys():
                self.kg_rel['is located at'] = dict()
            else:
                diff = self.compare_loc_record(self.kg_rel['is located at'], self.location_record)
                print('diff ', diff )
                print('self.kg_change',self.kg_change)
                # print('self.locations', self.kg_rel['is located at'])
                if diff: # and self.kg_change:
                    for d in diff:
                        # exist_bool = False
                        # i = 0
                        # while i < len(self.kg_change):
                        #
                        #     cha = self.kg_change[i]
                        #     if cha[0] == d[0] and cha[1] == d[1]:
                        #         self.kg_change[i][-1][-1] = d[-1][-1]
                        #
                        #         if self.kg_change[i][-1][0] == self.kg_change[i][-1][-1]:
                        #             self.kg_change.pop(i)  # The novelty change back~ No need now
                        #         else:
                        #             i += 1
                        #
                        #         exist_bool = True
                        #
                        # if exist_bool == False:
                        self.kg_change.append(d)

            # for loc in self.location_record:
            #     self.kg_rel['is located at'][loc] = self.location_record[loc]

            self.kg_rel['is located at'] = self.location_record.copy()
            self.location_record.clear()
        if self.kg_change_bool or self.update_num == self.update_interval or self.update_num in [1,2,3,4,5]:
            self.build_matrix_dict()
            self.sparse_matrix = self.dict_to_matrix()
            self.save_matrix()
            self.kg_change_bool = False  # Reset new_kg_tuple
             # Update history while only detect rule change after simulating 100 games
        if self.dice.novelty != self.dice_novelty:
            self.kg_change += ['Dice', self.dice.novelty[0]]
            self.dice_novelty = self.dice.novelty[:]

        if self.card_board.novelty != self.card_board_novelty:
            self.kg_change += self.card_board.novelty
            self.card_board_novelty = self.card_board_novelty[:]

        if self.dice.novelty or self.kg_change or self.card_board.novelty:
            return self.dice.type_record, self.kg_change, self.card_board.novelty
        else:
            return None

    def build_kg_text(self, text, level='sub',use_hash=False):
        """
        Use a logging sentence to build or add or update kg
        * _Check_
                * Check if the game rule **exists**, if yes, just ignore it.
                * Check if the game rule is about **location**.
                * Otherwise, add the game rule hash value to kg_set, so it will be easy to detect it in the future.
        * _Add_
            * Annotate the text with nlp server and get a tuple (sub, rel, obj).
            * If location related, add location record to self.location_record, a dict.
            * Add game rule to KG by ***self.kg_add()***
                * If we have this sub in this rel before, means kg changed, return True! If not, just add this to the kg graph.
                * After adding the new game rule to the big **self.kg_rel**, we also call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
            * When adding the game rule, if we see contradiction, we will call ***self.kg_update()*** to update the knowledge graph and return difference. PS: we did not update the game rule when there is contradiction in *self.kg_add()*.
                * Update **self.kg_rel_diff** to record the difference.
                * Update knowledge graph, **self.kg_rel**
                * Call *self.update_new_kg_tuple()* to put this new rule in another dict, **self.new_kg_tuple**.
                * return the difference: [sub, rel, old-obj, new-obj]
        * _Return_
            * Return the difference with a list of lists, [[sub, rel, old-obj, new-obj], [sub, rel, old-obj, new-obj],...]

        :param text: string. One sentence from logging info
        :param level: string. 'sub' or 'rel'. level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: list. Return the difference with a list of lists, [[sub, rel, old-obj, new-obj], [sub, rel, old-obj, new-obj],...]
        """

        diff = []
        diff_set = set()
        text = text.strip()
        # Add history of dice => Novelty 1 - Dice
        if 'Dice' in text and '[' in text:
            self.text_dice_num += 1
            dice_list = list(map(lambda x: int(x), text[text.index('[') + 1 : text.index(']')].split(',')))  # i.e. [2,5]
            self.dice.record_history_new_dice(dice_list)
            return diff

        # if 'Card' in text:
        #     self.text_card_num += 1
        #     card_name = text[text.index('>')+2:]
        #     pack = text.split(' ')[0]
        #     if pack != 'Chance':
        #         pack = 'Community-Chest'
        #     self.card.record_history_new_card(card_name, pack)
        #     return diff

        '''
        Except dice, then record and check game rule => Novelty 1 - Card and Novelty 2
        * _Check_
            * Check if the game rule **exists**, if yes, just ignore it.
            * Check if the game rule is about **location**.
            * Otherwise, add the game rule hash value to kg_set, so it will be easy to detect it in the future.
        '''

        if use_hash:
            triple_hash = hash(text)
            if triple_hash in self.kg_set and 'locate' not in text:  # Record this rule previously, just skip
                return diff
            elif triple_hash not in self.kg_set and 'locate' not in text:
                self.kg_set.add(triple_hash)  # Add to set for checking faster later on

        # Annotate the text with nlp server and get a tuple (sub, rel, obj)
        entity_relations = self.annotate(text, simple_format=True)

        for er in entity_relations:  # er is a dict() containing sub, rel and obj
            if 'locate' in text:  # about location
                # print('text', text)
                # print('er', er)
                self.add_loc_history(er)  # after some rounds, we will compare the loc history in build_file
                return diff  # return no diff
            else:  # other rules
                kg_change_once, type_update = self.kg_add(er,level=level, use_hash=use_hash)  # kg_change_once is a bool, True means rule change
            if kg_change_once:  # Bool
                diff_once = self.kg_update(er, level=level, type=type_update) # find difference
                if diff_once:
                    diff.append(diff_once)
        return diff

    def add_loc_history(self, triple):
        """
        Add location info to subject => dict()
        :param triple: dict()
        :return: None
        """
        if triple['subject'] in self.location_record:
            if triple['object'] not in self.location_record[triple['subject']]:

                self.location_record[triple['subject']].append(triple['object'])
        else:
            self.location_record[triple['subject']] =[triple['object']]

    def compare_loc_record(self, total, new):
        """
        Compare two location dict()'s difference
        :param total: old big dict()
        :param new: new small dict()
        :return: diff : list().
        """
        diff = []
        for space in total.keys():
            if space in new:
                if sorted(total[space]) != sorted(new[space]):
                    diff.append([space, 'is located at', [sorted(total[space]), sorted(new[space])]])
        for space in new.keys():
            if space not in total.keys():
                diff.append([space, 'is located at', [[], sorted(new[space])]])
        return diff

    def generate_graphviz_graph_(self, text: str = '', png_filename: str = './out/graph.png', level:str = 'acc', kg_level='rel'):
        """
        Plot the knowledge graph with exsiting kg
       :param (str | unicode) text: raw text for the CoreNLPServer to parse
       :param (list | string) png_filename: list of annotators to use
       :param (str) level: control we plot the whole image all the local knowledge graph
       """
        entity_relations = self.annotate(text, simple_format=True)
        """digraph G {
        # a -> b [ label="a to b" ];
        # b -> c [ label="another label"];
        }"""
        if level == 'single':
            graph = list()
            graph.append('digraph {')
            for er in entity_relations:
                kg_change, type_update = self.kg_add(er)
                graph.append('"{}" -> "{}" [ label="{}" ];'.format(er['subject'], er['object'], er['relation']))
            graph.append('}')
        else:
            graph = list()
            graph.append('digraph {')
            if kg_level == 'rel':
                for rel in self.kg_rel.keys():
                    for sub in self.kg_rel[rel]:
                        graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_rel[rel][sub], rel))
            else:
                for sub in self.kg_sub.keys():
                    for rel in self.kg_sub[sub]:
                        graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_sub[sub][rel], rel))
            graph.append('}')

        output_dir = os.path.join('.', os.path.dirname(png_filename))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_dot = os.path.join(tempfile.gettempdir(), 'graph.dot')
        with open(out_dot, 'w') as output_file:
            output_file.writelines(graph)

        command = 'dot -Tpng {} -o {}'.format(out_dot, png_filename)
        dot_process = Popen(command, stdout=stderr, shell=True)
        dot_process.wait()
        assert not dot_process.returncode, 'ERROR: Call to dot exited with a non-zero code status.'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        self.client.stop()
        del os.environ['CORENLP_HOME']

    # def save_json(self, save_dict):
    #     '''
    #     Save kg dict to json file
    #     :param level:
    #     :return: None
    #     '''
    #     import json
    #     if save_level == 'total':
    #         if level == 'sub':
    #             with open(self.jsonfile, 'w') as f:
    #                 json.dump(self.kg_sub, f)
    #         else:
    #             with open(self.jsonfile, 'w') as f:
    #                 json.dump(self.kg_rel, f)


    # def read_json(self, level='sub'):
    #     '''
    #     Read kg dict file from json file
    #     :param level:
    #     :return: None
    #     '''
    #     import json
    #     with open(self.jsonfile, 'r') as f:
    #         if level == 'sub':
    #             self.kg_sub = json.load(f)
    #         else:
    #             self.kg_rel = json.load(f)

    #only kg_rel needs sparse matrix
    def build_matrix_dict(self):
        '''
        build a dict for building sparse matrix
        names - price - price_1_house - rents....
        '''
        for i , rel in enumerate(self.relations_matrix):  # ['is priced at', ...]
            if rel in self.kg_rel.keys():
                for sub in self.kg_rel[rel].keys():
                    if sub in self.sparse_matrix_dict['nodes_number']:
                        sub_id = self.sparse_matrix_dict['nodes_number'][sub]
                        if type(sub_id) == int:
                            sub_id = [sub_id]
                        for number_sub in sub_id:
                            # Adjust for more rel/edges: TODO
                            # if i == 0:
                            number_obj = self.sparse_matrix_dict['nodes_number'][
                                self.relations_node[i] + str(self.hash_money(int(self.kg_rel[rel][sub]),i))]
                            # else:
                            #     if len(self.kg_rel[rel][sub]) > 1:
                            #         number_obj = []
                            #         for location in self.kg_rel[rel][sub]:
                            #             if self.relations_node[i] + str(location) in self.sparse_matrix_dict['nodes_number']:
                            #                 number_obj.append(int(self.sparse_matrix_dict['nodes_number'][self.relations_node[i] + str(location)]))
                            #     else:
                            #         number_obj = int(self.sparse_matrix_dict['nodes_number'][
                            #             self.relations_node[i] + str(self.kg_rel[rel][sub][0])])

                            self.sparse_matrix_dict['out'][rel][number_sub] = number_obj if type(number_obj) == list else [number_obj]

                            # if type(number_obj) == list:
                            #     for location in number_obj:
                            #         self.sparse_matrix_dict['in'][rel][location] = [int(number_sub)]
                            # else:
                            #     self.sparse_matrix_dict['in'][rel][number_obj] = [int(number_sub)]
        # print(self.sparse_matrix_dict['out'])
    def dict_to_matrix(self):
        self.sparse_matrix = []
        for i in range(self.node_number):  # node_id
            self.sparse_matrix.append([])
            for type in ['out']:
                for rel in self.relations_matrix:

                    node_1 = self.sparse_matrix_dict[type][rel][i] if i in self.sparse_matrix_dict[type][rel] else None
                    matrix_1 = [0 for j in range(self.node_number)]
                    if node_1:

                        if len(node_1) == 1:
                            matrix_1[node_1[0]] = 1 if node_1[0] > 0 else 0
                        else:
                            for loc in node_1:
                                matrix_1[loc] = 1

                    self.sparse_matrix[-1] += matrix_1

            # add color relationship like a and b share the same color so they will have a relationship
            name_i = self.sparse_matrix_dict['number_nodes'][i]
            matrix_1 = [0 for j in range(self.node_number)]
            if name_i in self.kg_rel['is colored as']:
                color_i = self.kg_rel['is colored as'][name_i]
                for sub, obj in self.kg_rel['is colored as'].items():
                    if obj == color_i and sub != name_i:
                        sub_number = self.sparse_matrix_dict['nodes_number'][sub]
                        matrix_1[sub_number] = 1
            self.sparse_matrix[-1] += matrix_1


        self.sparse_matrix = np.array(self.sparse_matrix)  # save as np array
        # for rel in self.action_name:
        #     self.sparse_matrix.append(csr_matrix((self.sparse_matrix_dict[rel]['data'], (self.sparse_matrix_dict[rel]['row'], self.sparse_matrix_dict[rel]['col'])), shape=(self.entity_num, self.entity_num)))
        return self.sparse_matrix

    def update_new_kg_tuple(self, triple):
        '''
        Update self.new_kg_tuple when there is new rule in kg
        :param triple: new kg rule tuple
        '''
        if triple['relation'] in self.new_kg_tuple.keys():
            pass
        else:
            self.new_kg_tuple[triple['relation']] = dict()

        self.new_kg_tuple[triple['relation']][triple['subject']] = triple['object']

    def save_matrix(self):
        '''
        Save sparse matrix of kg
        :return:
        '''
        print('self.matrix_file_path', self.matrix_file_path)
        np.save(self.matrix_file_path, self.sparse_matrix)
        # node_id = dict()
        # for node in self.sparse_matrix_dict['nodes_number']:
        #     if type(self.sparse_matrix_dict['nodes_number'][node]) == list:
        #         for i in range(len(self.sparse_matrix_dict['nodes_number'][node])):
        #             node_id[node + '_' + str(i)] = self.sparse_matrix_dict['nodes_number'][node][i]
        #     else:
        #         node_id[node] = self.sparse_matrix_dict['nodes_number'][node]
        #
        #
        # self.save_json(node_id, self.entity_file_path)
        self.save_json(self.sparse_matrix_dict['number_nodes'], self.entity_file_path)


    def save_vector(self):
        np.save(self.vector_file, self.kg_vector)

    def build_vector(self):
        '''
        Build the representation vector using knowledge graph
        '''
        num = 0
        for rel in self.relations_matrix:
            if rel in self.new_kg_tuple.keys():
                for sub in self.new_kg_tuple[rel].keys():
                    index_sub = int(self.board_name.index(sub))
                    obj = self.new_kg_tuple[rel][sub]
                    self.kg_vector[num][index_sub] = int(obj)
            num += 1

    # def record_history(self, name, history_dict):
    #     if name in history_dict.keys():
    #         history_dict[name]

class Novelty_Detection_Dice(History_Record):
    def __init__(self, config_file=None):
        #Novelty Detection
        # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
        self.upper_path = '/home/becky/Documents/Gatech_Agent_Eva_2'
        # if config_file == None:
        #     config_file = self.upper_path + '/monopoly_simulator_background/config.ini'

        config_data = ConfigParser()
        config_data.read(config_file)
        self.novelty_params = self.params_read(config_data, keys='novelty')
        self.new_dice = dict()
        self.dice = dict()
        self.percentage_var = self.novelty_params['percentage_var']
        self.num_dice = 0
        self.state_dice = []
        self.type_dice = []
        self.novelty = []
        self.type_record = [dict(), dict()]

        # For safety
        self.new_dice_temp = dict()
        self.temp_bool = False


    def run(self):
        '''
         = main function in this class
        :return: A list with tuples, including the dice novelty
        '''

        if self.temp_bool:
            novelty_temp = self.compare_dice_novelty(self.new_dice_temp, self.new_dice)
            novelty_temp_total = self.compare_dice_novelty(self.new_dice, self.dice)
            if novelty_temp_total == []:
                self.dice = self.add_new_to_total_dice(self.new_dice, self.dice)
                self.new_dice_temp.clear()
                self.new_dice.clear()
                self.temp_bool = False
                return []
            else:
                if novelty_temp:
                    self.temp_bool = True
                    self.new_dice_temp = self.new_dice.copy()
                    self.new_dice.clear()
                    return []
                else:
                    self.novelty.append(novelty_temp_total)
                    # print('temp', self.new_dice_temp, self.new_dice)
                    # self.type_record[1] = {'num': num_dice_new, 'state': state_dice_new, 'type': type_dice_new,
                    #                        'percentage': percentage_new}
                    # self.type_record[0] = {'num': num_dice, 'state': state_dice, 'type': type_dice, 'percentage': percentage}
                    self.dice.clear()
                    self.dice = self.add_new_to_total_dice(self.new_dice, self.dice)
                    self.dice = self.add_new_to_total_dice(self.new_dice_temp, self.dice)
                    self.temp_bool = False
                    self.new_dice.clear()
                    self.new_dice_temp.clear()
                    return novelty_temp_total

        else:
            novelty = self.compare_dice_novelty(self.new_dice, self.dice)

            if self.temp_bool:
                self.new_dice.clear()
                return []
            else:
                self.dice = self.add_new_to_total_dice(self.new_dice,self.dice)

            return novelty

    def record_history_new_dice(self, dice_list):
        '''
        Record the history of dice to new_dice dict
        :param dice_list (list):  a list indicating the dice from logging i.e. [2,3]
        :return: None
        '''
        for i, num in enumerate(dice_list):
            if i in self.new_dice.keys():
                if num in self.new_dice[i].keys():
                    self.new_dice[i][num] += 1
                else:
                    self.new_dice[i][num] = 1
            else:
                self.new_dice[i] = dict()
                self.new_dice[i][num] = 1

    def add_new_to_total_dice(self, new, total):
        for key in total.keys():
            if key in new.keys():
                total[key] = dict(Counter(total[key]) + Counter(new[key]))
        for key in new.keys():
            if key not in total.keys():
                total[key] = new[key]
        new.clear()
        return total

    def dice_evaluate(self, evaluated_dice_dict):
        '''
        Evaluate dice type, state, number
        :param evaluated_dice_dict (dict): put a dice history in dict
        :return: num_dice: # of dice used
                state_dice: state of each dice
                type_dice: dice are biased or uniform
        '''
        num_dice = len(evaluated_dice_dict.keys()) #int : 2
        state_dice = [] # [[1,2,3],[1,2]]
        type_dice = []
        percentages = []
        for key in evaluated_dice_dict.keys():
            state = list(map(lambda x: x[0], sorted(list(evaluated_dice_dict[key].items()), key=lambda x: x[0])))
            state_dice.append(state)
            nums = list(map(lambda x: x[1], sorted(list(evaluated_dice_dict[key].items()), key=lambda x: x[0])))
            percentage = [num / sum(nums) for num in nums]

            #Use KS-test to evaluate dice type:
            test_list = []
            for i, state_number in enumerate(state):
                test_list += [state_number for j in range(nums[i])]

            num_ks = 0
            p_value_com = 0
            while num_ks < 5:
                num_ks += 1
                test_distri = []
                for i, state_number in enumerate(state):
                    test_distri += [state_number for j in range(int(sum(nums)/len(state)))]

                p_value = stats.ks_2samp(np.array(test_list), np.array(test_distri)).pvalue
                p_value_com = max(p_value_com, p_value)
                if p_value_com > self.percentage_var:
                    break


            if p_value_com <= self.percentage_var:
                type_dice.append('Bias')
            else:
                type_dice.append('Uniform')
                percentage = [1/len(nums)]

            percentages.append(percentage)

        return num_dice, state_dice, type_dice, percentages

    def compare_dice_novelty(self, new_dice, dice):
        '''
        Dice Novelty Detection Type
        1. state
        2. type
        :return: bool. True means detecting novelty
        '''
        # print('self.new_dice',self.new_dice)
        # print('self.dice', self.dice)
        dice_novelty_list = []
        #Detect new state of dice. i.e. [1,2,3,4] => [1,2,3,4,5], we have a 5 now
        num_dice_new, state_dice_new, type_dice_new, percentage_new = self.dice_evaluate(new_dice)
        num_dice, state_dice, type_dice, percentage = self.dice_evaluate(dice)

        if num_dice_new != num_dice:
            dice_novelty_list.append(('Num', num_dice_new, num_dice))
        if state_dice_new != state_dice:
            dice_novelty_list.append(('State',state_dice_new, state_dice))
        if type_dice_new != type_dice:
            dice_novelty_list.append(('Type', type_dice_new, percentage_new, type_dice, percentage))

        if dice_novelty_list:
            # When U detect sth. new, do not tell the agent immediately
            if self.temp_bool == False:
                self.new_dice_temp = new_dice.copy()  # Record this and compare later
                self.temp_bool = True
                return []
            else:
                # self.novelty.append(dice_novelty_list)
                #
                # if dice == self.dice:
                #     self.type_record[1] = {'num': num_dice_new, 'state': state_dice_new, 'type': type_dice_new,
                #                            'percentage': percentage_new}
                #     self.type_record[0] = {'num': num_dice, 'state': state_dice, 'type': type_dice, 'percentage': percentage}
                # self.temp_bool = False

            # print('dice_novelty_list',dice_novelty_list)
                return dice_novelty_list

class Novelty_Detection_Card(History_Record):
    def __init__(self, config_file=None):
        # Novelty Detection
        # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
        self.upper_path = '/home/becky/Documents/Gatech_Agent_Eva_2'
        # if config_file == None:
        #     config_file = self.upper_path + '/monopoly_simulator_background/config.ini'

        config_data = ConfigParser()
        config_data.read(config_file)
        self.novelty_params = self.params_read(config_data, keys='novelty')
        self.card = dict()
        self.new_card = dict()

    def record_history_new_card(self,card_name, pack):
        if pack not in self.new_card.keys():
            self.new_card[pack] = dict()
        if card_name not in self.new_card[pack].keys():
            self.new_card[pack][card_name] = 1
        else:
            self.new_card[pack][card_name] += 1

import networkx as nx
import json
class Novelty_Detection_Card_Board(History_Record):
    def __init__(self, minimum_num=6):
        self.cards = dict()
        self.cards['chance'] = dict()
        self.cards['community_chest'] = dict()
        self.card_graph_state = nx.DiGraph()
        self.minimum_num = minimum_num
        self.novelty = []

    def ini_cards(self):
        card_board = self.read_json('/home/becky/Documents/Gatech_Agent_Eva_2/Evaluation_2/monopoly_game_schema_v1-2.json')['cards']
        for card_pack in ["community_chest", "chance"]:
            for card in card_board[card_pack]["card_states"]:
                name = card["name"]
                # update graph
                self.card_graph_state.add_node(name)

                for rel in card.keys():
                    if rel != 'name':
                        # update dict
                        self.cards[card_pack][name] = card
                        # add new nodes and edges
                        obj = card[rel]
                        try:
                            obj = int(obj)
                        except:
                            pass
                        self.card_graph_state.add_node(obj)
                        self.card_graph_state.add_edge(name, obj)
                        attrs = {(name, obj): {"attr": rel}}
                        nx.set_edge_attributes(self.card_graph_state, attrs)

    def read_card_board(self, card_board, game_num):
        for card_pack in ['picked_community_chest_card_details', 'picked_chance_card_details']:
            for name in card_board[card_pack]:
                # update dict
                if 'chance' in card_pack:
                    self.cards['chance'][name] = card_board[card_pack][name]
                else:
                    self.cards['community_chest'][name] = card_board[card_pack][name]
                # update graph
                if name not in self.card_graph_state.nodes:
                    self.card_graph_state.add_node(name)
                    # add new nodes and edges

                    for rel in card_board[card_pack][name]:
                        if rel != 'name':
                            obj = card_board[card_pack][name][rel]
                            try:
                                obj = int(obj)
                            except:
                                pass
                            self.card_graph_state.add_node(obj)
                            self.card_graph_state.add_edge(name, obj)
                            print('rel =>', rel)
                            attrs = {(name, obj): {"attr": rel}}
                            nx.set_edge_attributes(self.card_graph_state, attrs)


                    if game_num >= self.minimum_num:
                        print('new card node =>', name, 'game_num', game_num)

                        self.novelty.append(['new_card', name, dict(self.card_graph_state[name].items())])
                        print('card novelty', self.novelty[-1])

                else:
                    for rel in card_board[card_pack][name]:
                        if rel != 'name':
                            obj = card_board[card_pack][name][rel]
                            if obj not in self.card_graph_state[name]:
                                self.novelty.append(['card_change', name, dict(self.card_graph_state[name].items())])
                                print('card novelty', self.novelty[-1], 'game_num', game_num)
                                self.card_graph_state.add_node(obj)
                                self.card_graph_state.add_edge(name, obj)
                                attrs = {(name, obj): {"attr": rel}}
                                nx.set_edge_attributes(self.card_graph_state, attrs)

                    # update dict
                    if 'chance' in card_pack:
                        self.cards['chance'][name] = card_board[card_pack][name]
                    else:
                        self.cards['community_chest'][name] = card_board[card_pack][name]









