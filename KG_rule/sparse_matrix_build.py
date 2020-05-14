import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from configparser import ConfigParser

# class kg_to_matrix():
#     def __init__(self):
#         config_file = '/media/becky/GNOME-p3/monopoly_simulator/config.ini'
#         config_data = ConfigParser()
#         config_data.read(config_file)
#         self.hyperparams = self.params_read(config_data)
#         self.entity_num = self.hyperparams['entity_num']
#         self.action_num = self.hyperparams['action_num']
#         self.sparse_matrix = [csr_matrix((self.entity_num, self.entity_num), dtype=np.int8) for i in range(self.action_num)]
#         self.kg_sub = dict()
#         self.kg_rel = dict()
#         self.kg_path = self.hyperparams['kg_path']
#
#     def params_read(self, config_data):
#         params = {}
#         for key in config_data['matrix']:
#             v = eval(config_data['matrix'][key])
#             params[key] = v
#         # print('params',params)
#         return params
#
#     def read_kg(self, level):
#         import json
#         with open(self.kg_path, 'r') as f:
#             if level == 'sub':
#                 self.kg_sub = json.load(f)
#             else:
#                 self.kg_rel = json.load(f)
#
#     def build_matrix(self):


matrix_client = kg_to_matrix()
print(matrix_client.sparse_matrix)