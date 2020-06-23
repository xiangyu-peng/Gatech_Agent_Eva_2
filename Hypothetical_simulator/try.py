import numpy as np
import torch

import csv
# with open('eggs.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['a','b','c'])
# with open('eggs.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print(row)
#
# from gym.utils import seeding
# print(seeding.np_random(0))
# print(seeding.hash_seed(0 + 1) % 2 ** 31)

import pickle

import torch
torch.load('/media/becky/GNOME-p3/monopoly_simulator_background/weights/no_v3_lr_0.0001_#_66.pkl')



