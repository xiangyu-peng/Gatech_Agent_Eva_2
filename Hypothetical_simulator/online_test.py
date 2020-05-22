import sys
import os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
####################
from monopoly_simulator_background.vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt
import argparse


class Config:
    device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default=upper_path + '/monopoly_simulator_background/weights/v3_lr_0.001_#_18.pkl', type=str)
    parser.add_argument('--device_name', default='cuda:0', type=str)
    parser.add_argument('--num_t', default='cuda:0', type=str)
    parser.add_argument('--performance_count', default=10, type=int)

    args = parser.parse_args()
    params = vars(args)
    return params


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Hyp_Learner(object):
    """
    This class is separated into on-line learning and off-line learning.
    """
    #TODO:
    def __init__(self):
        args = parse_args()
        self.device = torch.device(args['device_name'])
        self.model = torch.load(args['model_path'])
        self.seed = args['seed']

        # game env/ novelty injection
        self.retrain_signal = False  # denotes if we will retrain the agent during next game
        self.performance_before_inject = [[], [], []]

        #performance of agents
        self.performance_count = args['performance_count']  # default is 10; we average the performance of 10 games

    def online_testing(self):
        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        env.set_kg(True)  # we need to use knowledge graph to detect the game rule change
        env.set_board()  # we use the original game-board in the beginning
        env.seed(self.seed)

        score = 0.0
        done = False
        win_num = 0
        avg_diff = 0

        with HiddenPrints():  # reset the env
            s, masked_actions = env.reset()

        num_test = 0  # count the # of games before novelty injection
        for _ in range(self.num_test):
            num_test += 1
            round_num_game = 0
            score_game = 0

            while not done:
                round_num_game += 1
                s = s.reshape(1, -1)
                s = torch.tensor(s, device=device).float()
                prob = model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                if masked_actions[a[0]] == 0:  # check whether the action is valid
                    a = [1]
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])
                s = s_prime
                score_game += r

            # s = s.cpu().numpy()[0]
            avg_diff += s[-2] - s[-1]
            score += score_game / round_num_game + 10 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0

            # Record the performance of the agent
            if num_test % self.performance_count == 0:
                self.performance_before_inject[0].append(round(score / self.performance_count, 3))     # score/ rewards
                self.performance_before_inject[0].append(round(win_num / self.performance_count, 3))   # winning rate
                self.performance_before_inject[0].append(round(avg_diff / self.performance_count, 3))  # difference of the cash at the end of game

            # Check the novelty of game
            if env.kg_change_output():
                self.retrain_signal = True
                self.performance_before_inject = [round(score / num_test, 3), round(win_num / num_test, 3), round(avg_diff / num_test, 3)]
                num_test = 0

                print(f"Step # :{step_idx}, avg score : {score / num_test:.3f}")
                print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
                print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")

        #TODO: Check the novelty

        env.close()


if __name__ == '__main__':
    hyp = Hyp_Learner()


# def add_vector_to_state(state, vector, device):
#     new_state = np.concatenate((vector.reshape(1, -1), state), axis=1)
#     return torch.tensor(new_state, device=device).float()
#
#
# def largest_prob(prob, masked_actions):
#     prob = prob.cpu().detach().numpy().reshape(-1, )
#     # Check if the action is valid
#     action_Invalid = True
#     largest_num = -1
#     while action_Invalid:
#         a = prob.argsort()[largest_num:][0]
#         action_Invalid = True if masked_actions[a] == 0 else False
#         a = [a]
#         largest_num -= 1
#     return a
#
#
# def test(step_idx, model, device, num_test, seed):
#     # Set the env for testing
#     env = gym.make('monopoly_simple-v1')
#     env.set_kg(False)
#     env.set_board()
#     env.seed(seed)
#
#     score = 0.0
#     done = False
#     win_num = 0
#     avg_diff = 0
#
#     with HiddenPrints():
#         s, masked_actions = env.reset()
#     # print('s',s)
#
#     for _ in range(num_test):
#
#         num_game = 0
#         score_game = 0
#
#         while not done:
#             num_game += 1
#             s = s.reshape(1, -1)
#             s = torch.tensor(s, device=device).float()
#
#             prob = model.actor(s)
#             # print(prob)
#             # break
#
#             # if num_game == 2:
#             #     print(prob, s)
#             #     s= s[0]
#             #     break
#             a = Categorical(prob).sample().cpu().numpy()  # substitute
#             if masked_actions[a[0]] == 0:
#                 a = [1]
#             with HiddenPrints():
#                 s_prime, r, done, masked_actions = env.step(a[0])
#             s = s_prime
#             score_game += r
#         # s = s.cpu().numpy()[0]
#         avg_diff += s[-2] - s[-1]
#         score += score_game / num_game + 10 * abs(abs(int(done) - 2) - 1)
#         win_num += abs(abs(int(done) - 2) - 1)
#         done = 0
#
#     print(f"Step # :{step_idx}, avg score : {score / num_test:.3f}")
#     print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
#     print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")
#
#     env.close()
#
#     return round(score / num_test, 5), round(win_num / num_test, 5), round(avg_diff / num_test, 5)
#
#
# if __name__ == '__main__':
#     config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
#     config_data = ConfigParser()
#     config_data.read(config_file)
#     # print('config_data.items', config_data.sections())
#     # Hyperparameters
#     n_train_processes = 1
#     learning_rate = 0.0002
#     update_interval = 5
#     gamma = 0.98
#     max_train_steps = 60000
#     PRINT_INTERVAL = update_interval * 10
#     config = Config()
#     config.hidden_state = 256
#     config.action_space = 90
#     save_dir = '/media/becky/GNOME-p3/monopoly_simulator_background'
#     # device = torch.device('cuda:0')
#     save_name = '/push_buy'
#     import os
#
#     # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     # torch.cuda.set_device(1)
#     device = torch.device("cpu")
#
#     #######################################
#     with HiddenPrints():
#         envs = ParallelEnv(n_train_processes)
#     # vector = np.load('/media/becky/GNOME-p3/KG-rule/vector.npy')
#     score_list, win_list, pie_list, diff_list, diff_neg_list = [], [], [], [], []
#     for seed in range(2, 3):
#         for i in range(1, 20):  # From 0(1) to 6(2) and interval is 5(3) => [0,5]
#             model_path = '/media/becky/GNOME-p3/monopoly_simulator_background/weights/cpu_v3_lr_0.001_#_' + str(
#                 i) + '.pkl'
#             model = torch.load(model_path)
#             print('i = ', i)
#             score, win, diff = test_v2(1, model, device, 500, seed=seed)
#             # score_list.append(score)
#             # win_list.append(win)
#             # pie_list.append(pie)
#             # diff_list.append(diff)
#             # diff_neg_list.append(diff_neg)
#
#     # #plot the figure
#     # x = np.linspace(0, 87, 30)
#     # plt.figure()
#     # plt.plot(x, score_list)
#     # plt.plot(x, win_list)
#     # plt.plot(x, pie_list)
#     # plt.plot(x, diff_list)
#     # # plt.plot(x, diff_neg_list)
#     # plt.legend(loc='best')
#     # plt.show()
#     # plt.savefig("Evaluate_A2C.png")