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

n = 2
sum_win = 0
while n < 3:
    done = 0
    env = gym.make('monopoly_simple-v1')
    env.seed(seed=n)
    # with HiddenPrints():
    s,mask = env.reset()

    while done == 0:
        # s, rew, done, info = env.step_nochange(0)
        # s, rew, done, info = env.step_after_nochange(0)
        # with HiddenPrints():
        s, rew, done, info = env.step(0)
    print(s)
    n += 1
    sum_win += done - 1
print(sum_win)