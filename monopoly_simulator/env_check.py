from vanilla_A2C import *
from configparser import ConfigParser
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

env = gym.make('monopoly_simple-v1')
print('seeds =====>', env.seed(seed=0))
with HiddenPrints():
    s, masked = env.reset()
print('s =====> reset', s)
print('masked', masked)
print('======================step======================')
with HiddenPrints():
    s = env.step(0) #buy
print('======================step======================')
print('s =====>', s)
# s = env.step(25) #mortgage
# print('======================step======================')
# print('s =====>', s)
# s = env.step(53) #free-mortgage
# print('======================step======================')
# print('s =====>', s)
# s = env.step(79) #do nothing
# print('======================step======================')
# print('s =====>', s)