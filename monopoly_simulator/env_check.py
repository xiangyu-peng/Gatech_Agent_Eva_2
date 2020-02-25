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



done = False
env = gym.make('monopoly_simple-v1')
env.seed(seed=5)
with HiddenPrints():
    s,mask = env.reset()
with HiddenPrints():
    while done == False:
        s, rew, done,info = env.step(-1)
print('s =====> reset', s)
print('done',done)
print('env.seeds', env.seeds)
env.close()

done = False
# env = gym.make('monopoly_simple-v1')
# env.seed(seed=6)
with HiddenPrints():
    s,mask = env.reset()
with HiddenPrints():
    while done == False:
        s, rew, done,info = env.step(-1)
print('s =====> reset', s)
print('done',done)
print('env.seeds', env.seeds)
env.close()
# done = False
# with HiddenPrints():
#     while done == False:
#         s, rew, done,info = env.step(0)
# print('s =====> reset', s)




# print('masked', masked)
# print('======================step======================')
# with HiddenPrints():
#     s = env.step(0) #buy
# print('======================step======================')
# print('s =====>', s)
# s = env.step(25) #mortgage
# print('======================step======================')
# print('s =====>', s)
# s = env.step(53) #free-mortgage
# print('======================step======================')
# print('s =====>', s)
# s = env.step(79) #do nothing
# print('======================step======================')
# print('s =====>', s)