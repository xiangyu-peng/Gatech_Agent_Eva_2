#####Evaluation#####
import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator','')
upper_path_eva = upper_path + '/Evaluation/monopoly_simulator'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
#####################################

import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr
from zipfile import ZipFile
from configparser import ConfigParser
import sys
import os
from monopoly_simulator_background.gameplay_tf import set_up_board

import wget
import numpy as np
from monopoly_simulator_background.interface import Interface
from scipy.sparse import csr_matrix, load_npz, save_npz
from configparser import ConfigParser
from collections import Counter
import random
from scipy import stats

from monopoly_simulator_background.simple_background_agent_becky_p1 import P1Agent
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent

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
        '''
        Save kg dict to json file
        For kg save_dict = self.kg_rel
        For kg save_path = self.jsonfile
        :param level:
        :return: None
        '''

        # if type == 'json':
        #     import json
        #     with open(save_path, 'w') as f:
        #         json.dump(save_dict, f)
        # if type == 'pickle':
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def read_json(self, load_dict, load_path):
        '''
        Read kg dict file from json file
        :param level:
        :return: None
        '''

        load_dict = dict()
        # if type == 'json':
        #     with open(load_path, 'r') as f:
        #         load_dict = json.load(f)

        # if type == 'pickle':
        with open(load_path, 'rb')as f:
            load_dict = pickle.load(f)

        return load_dict





class KG_OpenIE(History_Record):
    def __init__(self, gameboard,\
                 core_nlp_version: str = '2018-10-05', config_file=None):
        self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
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
        from stanfordnlp.server import CoreNLPClient

        # For generating kg
        if config_file == None:
            config_file = self.upper_path + '/monopoly_simulator_background/config.ini'
        config_data = ConfigParser()
        config_data.read(config_file)
        self.params = self.params_read(config_data, keys='kg')
        self.jsonfile = self.upper_path + self.params['jsonfile']
        self.client = CoreNLPClient(annotators=['openie'], memory='8G')
        self.relations = ['priced', 'rented', 'located', 'colored', 'classified', 'away', 'type', 'cost', 'direct', 'mortgaged', 'increment']
        self.relations_full = ['is priced at', 'is located at', 'is rented-0-house at'] #, 'is colored as', 'is classified as']
        self.kg_rel = dict()  # the total kg rule for "rel" KG
        self.kg_sub = dict()  # the total kg rule for "sub" KG
        self.kg_set = set()   # the set recording the kg rule for searching if the rule exists quickly
        self.kg_rel_diff = dict()  # the dict() to record the rule change
        self.kg_sub_diff = dict()

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

        self.sparse_matrix_dict = self.build_empty_matrix_dict()
        self.matrix_folder = self.upper_path + self.matrix_params['matrix_folder']
        self.kg_vector = np.zeros([len(self.relations_full), len(self.board_name)])
        self.vector_file = self.upper_path + self.matrix_params['vector_file']

        # Dice Novelty
        self.dice = Novelty_Detection_Dice()
        self.text_dice_num = 0
        self.dice_novelty = []
        # self.text_card_num = 0
        # self.card = Novelty_Detection_Card()

    def set_gameboard(self, gameboard):
        self.gameboard = gameboard
        self.board_name = [i.name for i in gameboard['location_sequence']]
        for i, name in enumerate(self.board_name):
            self.board_name[i] = '-'.join(name.split(' '))



    def build_empty_matrix_dict(self):
        sparse_matrix_dict = dict()
        for rel in self.action_name:
            sparse_matrix_dict[rel] = dict()
            sparse_matrix_dict[rel]['row'] = []
            sparse_matrix_dict[rel]['col'] = []
            sparse_matrix_dict[rel]['data'] = []
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
                        if rel in triple['relation']:

                            #map B&0-Railroad
                            if 'B-' in triple['subject']:
                                triple['subject'] = 'B&O-Railroad'
                            if 'B-' in triple['object']:
                                triple['object'] = 'B&O-Railroad'

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

    def kg_update(self, triple, level='sub'):
        """
        After detecting rule change, update kg and also return the diff of kg
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
                self.kg_rel_diff[triple['relation']][triple['subject']] = [self.kg_rel[triple['relation']][triple['subject']]]

            if triple['subject'] not in self.kg_rel_diff[triple['relation']].keys():
                self.kg_rel_diff[triple['relation']][triple['subject']] = [self.kg_rel[triple['relation']][triple['subject']]]

            self.kg_rel[triple['relation']][triple['subject']] = triple['object']  # update kg
            self.update_new_kg_tuple(triple)

            self.kg_rel_diff[triple['relation']][triple['subject']].append(triple['object'])

            return [triple['subject'], triple['relation'], self.kg_rel_diff[triple['relation']][triple['subject']]]


    def kg_add(self, triple, level='sub', use_hash=False):
        """
        Add new triple (sub, rel, obj) to knowledge graph
        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed, False means not changed or no existing rule, and add a rule
        """
        if level == 'sub':
            if triple['subject'] in self.kg_sub.keys():
                if triple['relation'] in self.kg_sub[triple['subject']].keys():

                    return True if self.kg_sub[triple['subject']][triple['relation']] != triple['object'] else False
                else:
                    self.kg_sub[triple['subject']][triple['relation']] = triple['object']

            else:
                self.kg_sub[triple['subject']] = dict()
                self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            return False

        else:  # level = 'rel'
            if triple['relation'] in self.kg_rel.keys():
                if triple['subject'] in self.kg_rel[triple['relation']].keys():
                    if use_hash:  # If we use hash, there is no need to check here, return True directly is ok
                        if type(self.kg_rel[triple['relation']][triple['subject']]) == list:  # location
                            return True if triple['object'] not in self.kg_rel[triple['relation']][triple['subject']] else False
                        return True
                    else:
                        return True if self.kg_rel[triple['relation']][triple['subject']] != triple['object'] else False
                    # True here means there is a change in the rule, False means stay the same.


                else:
                    self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                    self.update_new_kg_tuple(triple)

            else:
                self.kg_rel[triple['relation']] = dict()
                self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                self.update_new_kg_tuple(triple)
            return False

    def build_kg_file(self, file_name, level='sub', use_hash=False):
        """
        Give the logging file and add kg to the existing kg
        :param file_name: logger info file path
        :param level: 'sub' or 'rel'
        :param use_hash: bool. Make the check of existing rule much faster
        :return:
        """

        file = open(file_name, 'r')

        for line in file:
            kg_change = self.build_kg_text(line, level=level, use_hash=use_hash)

            # Check dice novelty
            if self.text_dice_num == self.detection_num:
                self.dice.dice = self.dice.add_new_to_total_dice(self.dice.new_dice, self.dice.dice)
            if self.text_dice_num > self.detection_num and self.text_dice_num % self.history_update_interval == 0:
                self.dice.run()
                self.text_dice_num = 1 + self.detection_num

            # Check card novelty

            if kg_change:
                if kg_change not in self.kg_change:
                    self.kg_change += kg_change

        self.update_num += 1


        # If there is any update or new relationships in kg, will update in the matrix

        if self.update_num % self.update_interval == 0:

            #solve differnce of locations
            if 'is located at' not in self.kg_rel.keys():
                self.kg_rel['is located at'] = dict()
            else:
                diff = self.compare_loc_record(self.kg_rel['is located at'], self.location_record)
                if diff:
                    for d in diff:
                        exist_bool = False
                        for i, cha in enumerate(self.kg_change):
                            if cha[0] == d[0] and cha[1] == d[1]:
                                self.kg_change[i][-1][-1] = d[-1][-1]

                                if self.kg_change[i][-1][0] == self.kg_change[i][-1][-1]:
                                    self.kg_change.pop(i)  # The novelty change back~ No need now

                                exist_bool = True
                        if exist_bool == False:
                            self.kg_change.append(d)

            self.kg_rel['is located at'] = self.location_record.copy()
            self.location_record.clear()
            if self.new_kg_tuple:
                self.build_matrix_dict()
                self.dict_to_matrix()
                self.build_vector()
                self.new_kg_tuple = dict()  # Reset new_kg_tuple
                 # Update history while only detect rule change after simulating 100 games

        if self.dice.novelty != self.dice_novelty:
            self.kg_change += self.dice.novelty
            self.dice_novelty = self.dice.novelty[:]

        if self.dice.novelty or self.kg_change:
            return self.dice.type_record, self.kg_change
        else:
            return None

    def build_kg_text(self, text, level='sub',use_hash=False):
        """
        Use a logging sentence to build or add to kg
        :param text (string): One sentence from logging info
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed
        """

        diff = []
        diff_set = set()

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


        # Except dice, then record and check game rule => Novelty 1 - Card and Novelty 2
        if use_hash:
            triple_hash = hash(text)
            if triple_hash in self.kg_set and 'locate' not in text:  # Record this rule previously, just skip
                return diff
            else:
                self.kg_set.add(triple_hash)  # Add to set for checking faster later on

        entity_relations = self.annotate(text, simple_format=True)

        for er in entity_relations:  # er is a dict() containing sub, rel and obj
            if 'locate' in text:
                kg_change_once = self.add_loc_history(er)
                return diff
            else:
                kg_change_once = self.kg_add(er,level=level, use_hash=use_hash)  # kg_change_once is a bool, True means rule change

            if kg_change_once:
                diff_once = self.kg_update(er, level=level)
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
                kg_change = self.kg_add(er)
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
        '''
        for rel in self.action_name:
            if rel in self.new_kg_tuple.keys():
                for sub in self.new_kg_tuple[rel].keys():
                    index_sub = self.board_name.index(sub)
                    index_obj = self.board_name.index(self.new_kg_tuple[rel][sub])
                    self.sparse_matrix_dict[rel]['row'].append(index_sub)
                    self.sparse_matrix_dict[rel]['col'].append(index_obj)
                    self.sparse_matrix_dict[rel]['data'].append(1)

    def dict_to_matrix(self):
        self.sparse_matrix = []
        for rel in self.action_name:
            self.sparse_matrix.append(csr_matrix((self.sparse_matrix_dict[rel]['data'], (self.sparse_matrix_dict[rel]['row'], self.sparse_matrix_dict[rel]['col'])), shape=(self.entity_num, self.entity_num)))

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
        num = 0
        for rel in self.action_name:
            save_npz(self.matrix_folder + '/' + str(rel) + '.npz', self.sparse_matrix[num])
            num += 1

    def save_vector(self):
        np.save(self.vector_file, self.kg_vector)

    def build_vector(self):
        '''
        Build the representation vector using knowledge graph
        '''
        num = 0
        for rel in self.relations_full:
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
        self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
        if config_file == None:
            config_file = self.upper_path + '/monopoly_simulator_background/config.ini'

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
        self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
        if config_file == None:
            config_file = self.upper_path + '/monopoly_simulator_background/config.ini'
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






if __name__ == '__main__':

    # #the following code only for debugging, ignore it!!!
    # #############gameboard######
    player_decision_agents = dict()
    player_decision_agents['player_1'] = P1Agent()

    # Assign the beckground agents to the players
    name_num = 1
    while name_num < 4:
        name_num += 1
        player_decision_agents['player_' + str(name_num)] = Agent(**background_agent_v3.decision_agent_methods)

    gameboard_initial = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-1.json',
                                     player_decision_agents, num_active_players=4)

    client = KG_OpenIE(gameboard_initial)
    client.annotate('I love U')
    # client.build_kg_file('/media/becky/GNOME-p3/KG-rule/game_log.txt', level='rel', use_hash=True)

    #
    # file_name='/media/becky/GNOME-p3/KG-rule/test.txt'
    #
    # for i in range(100):
    #     file = open("/media/becky/GNOME-p3/KG-rule/test.txt", "w")
    #     for j in range(500):
    #         l = []
    #         l.append(random.randint(1,6))
    #         l.append(random.randint(1,6))
    #         file.write('Dice => ' + str(l) +' \n')
    #     file.close()
    #     client.build_kg_file(file_name, level='rel', use_hash=True)
    #
    #
    # for i in range(100):
    #     file = open("/media/becky/GNOME-p3/KG-rule/test.txt", "w")
    #     for j in range(300):
    #
    #         l = []
    #         l.append(random.randint(1,6))
    #         l.append(random.randint(1,3) + random.randint(0,2))
    #         file.write('Dice => ' + str(l) +' \n')
    #     # file.close()
    #     client.build_kg_file(file_name, level='rel', use_hash=True)
    #
    #     # print(client.kg_change)
    # print(client.dice.type_record)
    # print(client.dice.new_dice)
    # print(client.kg_change)



    # client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')


    # dice = Novelty_Detection()
    # for i in range(10000):
    #     l = []
    #     l.append(random.randint(1,3))
    #     l.append(random.randint(1,2)+random.randint(1,2))
    #     dice.record_history_new_dice(l)
    # print(dice.dice_evaluate(dice.new_dice))
    # print(dice.new_dice)
    # dice.add_new_to_total_dice()
    # for i in range(1000):
    #     l = []
    #     l.append(random.randint(1,3))
    #     l.append(random.randint(2,4))
    #     dice.record_history_new_dice(l)
    # print(dice.new_dice)
    # print(dice.dice_evaluate(dice.new_dice))
    # print(dice.compare_dice_novelty())

#     import time
#     start = time.time()
#     file='/media/becky/GNOME-p3/KG-rule/game_log.txt'
#     log_file = open(file,'r')
#
# #############gameboard######
#     player_decision_agents = dict()
#     player_decision_agents['player_1'] = P1Agent()
#
#     # Assign the beckground agents to the players
#     name_num = 1
#     while name_num < 4:
#         name_num += 1
#         player_decision_agents['player_' + str(name_num)] = P2Agent_v2()
#
#     gameboard_initial = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-1.json',
#                                           player_decision_agents, num_active_players=4)
#
#     client = KG_OpenIE(gameboard_initial)
#
#     # client.read_json(level='rel')
#     # client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')
#     i = 0
#     while i < 15:
#         i += 1
#         client.build_kg_file(file, level='rel', use_hash=True)
#
#     # file = '/media/becky/GNOME-p3/KG-rule/game_log_2.txt'
#     # log_file = open(file, 'r')
#
#     # i = 0
#     # while i < 15:
#     #     i += 1
#     #     client.build_kg_file(file, level='rel', use_hash=True)
#     #
#     #
#     # file = '/media/becky/GNOME-p3/KG-rule/game_log.txt'
#     # log_file = open(file, 'r')
#     #
#     # i = 0
#     # while i < 20:
#     #     i += 1
#     #     client.build_kg_file(file, level='rel', use_hash=True)
#     print(client.kg_rel)
#     print(client.kg_change)
#     client.dict_to_matrix()
#     # client.save_matrix()
#     print(client.kg_vector)
#     # client.save_vector()




    #
    # # client.read_json(level='rel')
    # for line in log_file:
    #     kg_change = client.build_kg(line,level='rel',use_hash=True)
    # # client.save_json(level='rel')
    # # client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')
    #
    # # print(client.kg_rel.keys())
    # # line = 'Vermont-Avenue is colored as SkyBlue'
    # # kg_change = client.build_kg(line,level='sub')
    # print(client.kg_rel_diff)
    # end = time.time()
    # print(str(end-start))

    # row = np.array([0, 0, 1, 2, 2, 1])
    # col = np.array([0, 2, 2, 0, 1, 2])
    # data = np.array([1, 2, 3, 4, 5, 6])
    # a = csr_matrix((data, (row, col)), shape=(3, 3))
    # print('a',a)
    # save_npz('/media/becky/GNOME-p3/KG-rule/matrix_rule/a.npz',a)
    # sparse_matrix = load_npz('/media/becky/GNOME-p3/KG-rule/matrix_rule/a.npz')
    # print('sparse_matrix',sparse_matrix)
    # import pickle
    # a = set()
    # a.add(1)
    # a.add(2)
    # a.add(3)
    # file = open('pickle_example.pickle', 'wb')
    # pickle.dump(a, file)
    # file.close()
    # with open('/media/becky/GNOME-p3/KG-rule/json_kg.pickle', 'rb')as file:
    #     a_dict1 = pickle.load(file)
    # print(a_dict1)

    # import json
    # a = dict()
    # a['love'] = 'U'
    # with open('/media/becky/GNOME-p3/KG-rule/json_kg.json', 'w') as f:
    #     json.dump(a, f)
    #
    # with open('/media/becky/GNOME-p3/KG-rule/json_kg.json', 'r') as f:
    #     b = json.load(f)
    # print(b)
