import os
import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr
from zipfile import ZipFile
from configparser import ConfigParser
import sys
curr_path = os.getcwd()
curr_path = curr_path.replace("/KG-rule", "")
curr_path = curr_path.replace("/env", "")
curr_path = curr_path.replace("/monopoly_simulator", "")
sys.path.append(curr_path + '/env')
import wget
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from configparser import ConfigParser


class KG_OpenIE():
    def __init__(self, core_nlp_version: str = '2018-10-05', config_file='/media/becky/GNOME-p3/monopoly_simulator/config.ini'):
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

        #for generating kg
        config_data = ConfigParser()
        config_data.read(config_file)
        self.params = self.params_read(config_data, keys='kg')
        self.jsonfile = self.params['jsonfile']
        self.client = CoreNLPClient(annotators=['openie'], memory='8G')
        self.relations = ['priced', 'rented', 'located', 'colored', 'classified', 'away']
        self.kg_rel = dict()
        self.kg_sub = dict()
        self.kg_set = set()
        self.kg_rel_diff = dict()
        self.kg_sub_diff = dict()
        self.kg_introduced = False
        self.new_kg_tuple = dict()
        self.update_num  = 0

        #for kg to matrix
        self.matrix_params = self.params_read(config_data, keys='matrix')
        self.entity_num = self.matrix_params['entity_num']
        self.action_num = self.matrix_params['action_num']
        self.sparse_matrix = []
        self.action_name = ['is ' + str(i) +'-step away from' for i in range(1,41)]
        self.board_name = ['Go','Mediterranean Avenue', 'Community Chest-One',
                'Baltic Avenue', 'Income Tax', 'Reading Railroad', 'Oriental-Avenue',
                'Chance-One', 'Vermont Avenue', 'Connecticut Avenue', 'In Jail/Just Visiting',
                'St. Charles Place', 'Electric Company', 'States Avenue', 'Virginia Avenue',
                'Pennsylvania Railroad', 'St. James Place', 'Community Chest-Two', 'Tennessee Avenue',
                'New York Avenue', 'Free Parking', 'Kentucky Avenue', 'Chance-Two', 'Indiana Avenue',
                'Illinois Avenue', 'B&O Railroad', 'Atlantic Avenue', 'Ventnor Avenue',
                'Water Works', 'Marvin Gardens', 'Go-to-Jail', 'Pacific Avenue', 'North Carolina Avenue',
                'Community Chest-Three', 'Pennsylvania Avenue', 'Short Line', 'Chance-Three', 'Park Place',
                                        'Luxury Tax', 'Boardwalk']

        self.sparse_matrix_dict = self.build_empty_matrix_dict()
        self.matrix_folder = self.matrix_params['matrix_folder']


    def build_empty_matrix_dict(self):
        sparse_matrix_dict = dict()
        for rel in self.action_name:
            sparse_matrix_dict[rel] = dict()
            sparse_matrix_dict[rel]['row'] = []
            sparse_matrix_dict[rel]['col'] = []
            sparse_matrix_dict[rel]['data'] = []
        return sparse_matrix_dict

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

    def annotate(self, text: str, properties_key: str = None, properties: dict = None, simple_format: bool = True):
        """
        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (str) properties_key: key into properties cache for the client
        :param (dict) properties: additional request properties (written on top of defaults)
        :param (bool) simple_format: whether to return the full format of CoreNLP or a simple dict.
        :return: Depending on simple_format: full or simpler format of triples <subject, relation, object>.
        """
        # https://stanfordnlp.github.io/CoreNLP/openie.html
        core_nlp_output = self.client.annotate(text=text, annotators=['openie'], output_format='json',
                                               properties_key=properties_key, properties=properties)
        if simple_format:
            triples = []
            for sentence in core_nlp_output['sentences']:
                for triple in sentence['openie']:

                    for rel in self.relations:
                        if rel in triple['relation']:
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
        '''
        After detecting rule change, update kg and also return the diff of kg
        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: A tuple (sub, rel, diff)
        '''
        if level == 'sub':
            if triple['subject'] not in self.kg_sub_diff.keys():
                self.kg_sub_diff[triple['subject']] = dict()
                self.kg_sub_diff[triple['subject']][triple['relation']] = [self.kg_sub[triple['subject']][triple['relation']]]
            self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            self.kg_sub_diff[triple['subject']][triple['relation']].append(triple['object'])
            return (triple['subject'],triple['relation'],self.kg_sub_diff[triple['subject']][triple['relation']])
        else:
            if triple['relation'] not in self.kg_rel_diff.keys():
                self.kg_rel_diff[triple['relation']] = dict()
                self.kg_rel_diff[triple['relation']][triple['subject']] = [self.kg_rel[triple['relation']][triple['subject']]]
            self.kg_rel[triple['relation']][triple['subject']] = triple['object']
            self.update_new_kg_tuple(triple)
            self.kg_rel_diff[triple['relation']][triple['subject']].append(triple['object'])
            return (triple['subject'], triple['relation'], self.kg_rel_diff[triple['relation']][triple['subject']])


    def kg_add(self, triple, level='sub'):
        '''
        Add new triple (sub, rel, obj) to knowledge graph
        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed, False means not changed or no existing rule, and add a rule
    '''
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

        else: #level = 'rel'
            if triple['relation'] in self.kg_rel.keys():
                if triple['subject'] in self.kg_rel[triple['relation']].keys():
                    return True if self.kg_rel[triple['relation']][triple['subject']] != triple['object'] else False
                else:
                    self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                    self.update_new_kg_tuple(triple)


            else:
                self.kg_rel[triple['relation']] = dict()
                self.kg_rel[triple['relation']][triple['subject']] = triple['object']
                self.update_new_kg_tuple(triple)
            return False

    def build_kg_file(self, file_name, level='sub', use_hash=False, update_interval=1):
        file = open(file_name, 'r')
        for line in file:
            kg_change = self.build_kg_text(line, level=level, use_hash=use_hash)

        self.update_num += 1
        #if there is any update or new relationships in kg, will update in the matrix
        if self.update_num % update_interval == 0:
            if self.new_kg_tuple:
                self.build_matrix_dict()
                self.dict_to_matrix()
                self.new_kg_tuple = dict()

    def build_kg_text(self, text, level='sub',use_hash=False):
        '''
        Use a logging sentence to build or add to kg
        :param text (string): One sentence from logging info
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed
        '''
        diff = []
        if use_hash:
            triple_hash = hash(text)
            if triple_hash in self.kg_set:
                return diff
            else:
                self.kg_set.add(triple_hash)
        entity_relations = self.annotate(text, simple_format=True)
        for er in entity_relations:
            kg_change_once = self.kg_add(er,level=level)
            if kg_change_once:
                diff.append(self.kg_update(er, level=level))

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

    def save_json(self, level='sub'):
        '''
        Save kg dict to json file
        :param level:
        :return: None
        '''
        import json
        if level == 'sub':
            with open(self.jsonfile, 'w') as f:
                json.dump(self.kg_sub, f)
        else:
            with open(self.jsonfile, 'w') as f:
                json.dump(self.kg_rel, f)

    def read_json(self, level='sub'):
        '''
        Read kg dict file from json file
        :param level:
        :return: None
        '''
        import json
        with open(self.jsonfile, 'r') as f:
            if level == 'sub':
                self.kg_sub = json.load(f)
            else:
                self.kg_rel = json.load(f)

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
        if triple['relation'] in self.new_kg_tuple.keys():
            pass
        else:
            self.new_kg_tuple[triple['relation']] = dict()
        self.new_kg_tuple[triple['relation']][triple['subject']] = triple['object']

    def save_matrix(self):
        num = 0
        for rel in self.action_name:
            save_npz(self.matrix_folder + '/' + str(rel) + '.npz', self.sparse_matrix[num])
            num += 1





# import time
# start = time.time()
file='/media/becky/GNOME-p3/monopoly_simulator/gameplay.log'
# log_file = open(file,'r')
client = KG_OpenIE()
client.build_kg_file(file, level='rel', use_hash=True, update_interval=1)
client.save_matrix()




#
# # client.read_json(level='rel')
# for line in log_file:
#     kg_change = client.build_kg(line,level='rel',use_hash=True)
# # client.save_json(level='rel')
# # client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')
#
# # print(client.kg_rel.keys())
# # line = 'Vermont Avenue is colored as SkyBlue'
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