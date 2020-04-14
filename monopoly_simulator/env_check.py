from vanilla_A2C import *
from configparser import ConfigParser
import os, sys
import time

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# n = 8
# sum_win = 0
# while n < 9:
#     done_num = 0
#     env = gym.make('monopoly_simple-v1')
#     env.seed(seed=n)
#     with HiddenPrints():
#         s,mask = env.reset()
#     done = 0
#     while done_num < 3:
#         # s, rew, done, info = env.step_nochange(0)
#         # s, rew, done, info = env.step_after_nochange(0)
#
#         with HiddenPrints():
#             s, rew, done, info = env.step_nochange(0)
#         # print('s', s, rew)
#         # print('info',info)
#         # break
#         with HiddenPrints():
#             s_later, rew_later, _, info = env.step_after_nochange(0)
#         # print('s =>', s)
#         # print('s_later', s_later)
#         if done_num ==1:
#             done_num+=1
#             print('s =>',s,rew)
#             print('s_later', s_later,rew_later)
#             print('info=>',info)
#             break
#         if done:
#             done_num += 1
#             print(s_later)
#
#
#     n += 1
#     sum_win += done - 1
# print(sum_win)
start = time.time()
n = 0
env = gym.make('monopoly_simple-v1')
env.seed(seed=n)
# with HiddenPrints():
s,mask = env.reset()
print(s,mask)
done_num = 0
done_num_total = 2
while done_num < done_num_total:
    with HiddenPrints():
        s, rew, done, info = env.step_nochange(0)
    print('s',s, rew, info)

    if done > 0:
        # print('Done')
        done_num += 1
        print('s', s, rew)

    # if done_num == done_num_total:
    #     break
    done_after = False
    while done_after == False:
        with HiddenPrints():
            s, rew, done_after, info = env.step_hyp(0)
        print('s', s, rew, info, done_after)

        with HiddenPrints():
            s, rew, done_after, info = env.step_hyp(1)
        print('s', s, rew, info, done_after)

    with HiddenPrints():
        s, rew, done_after, info = env.step_after_nochange(0)
    print('s', s, rew, info)
    print('===========')

    done_num += 1
    # print('s',s,rew)
end = time.time()
print('Run %d environments need %f' % (done_num_total, float(str(end-start))))
# s, rew, done, info = env.step_nochange(1)
# print('s',s, rew)
# s, rew, done_after, info = env.step_after_nochange(0)
# print('s',s,rew)

# s, rew, done, info = env.step_nochange(0)
# print('s',s, rew)
# s, rew, done, info = env.step_after_nochange(0)
# print('s',s,rew)