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
import logging
from Hypothetical_simulator.logging_info import log_file_create
logger = logging.getLogger('logging_info.online_testing')

class Config:
    device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default=upper_path + '/monopoly_simulator_background/weights/v3_lr_0.001_#_18.pkl', type=str)
    parser.add_argument('--device_name', default='cuda:0', type=str)
    parser.add_argument('--num_test', default=600, type=int)
    parser.add_argument('--performance_count', default=100, type=int)
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
    def __init__(self):
        args = parse_args()
        self.device = torch.device(args['device_name'])
        self.model = torch.load(args['model_path'])
        self.seed = args['seed']

        # game env/ novelty injection
        self.retrain_signal = False  # denotes if we will retrain the agent during next game
        self.performance_before_inject = [[], [], []]
        self.novelty_spaces = set()
        self.novelty = [[]]

        #performance of agents
        self.num_test = args['num_test']
        self.performance_count = args['performance_count']  # default is 10; we average the performance of 10 games

        # logger info
        self.logger = log_file_create(upper_path + '/Hypothetical_simulator/log_testing.log')

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
        done_reset_bool = False
        for _ in range(self.num_test):
            num_test += 1
            round_num_game, score_game = 0, 0
            retrain_signal_per_game = False  # define if we need to retrain tall the states after this state

            while not done:
                # print(round_num_game, s)
                round_num_game += 1
                s = s.reshape(1, -1)

                # check the novelty in the state
                # TODO: run hypothetical simulator
                # TODO: save all the game board before the novelty injection as well
                if self.novelty_spaces and not retrain_signal_per_game:
                    # Find the state, involving novelty
                    retrain_signal_per_game = self.find_novelty_state(env.output_interface(), s[0])
                    # Save each game board before the novelty
                    # if novelty is found in s, do not save, and begin retrain
                    if retrain_signal_per_game == False and done_reset_bool == False:
                        code = env.save_gameboard()
                        print(num_test, code)



                s = torch.tensor(s, device=self.device).float()
                prob = self.model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                if masked_actions[a[0]] == 0:  # check whether the action is valid
                    a = [1]
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])
                s = s_prime
                score_game += r
                done_reset_bool = False

            avg_diff += s[-2] - s[-1]
            score += score_game / round_num_game + 10 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0
            done_reset_bool = True

            # Record the performance of the agent
            if num_test % self.performance_count == 0:
                self.performance_before_inject[0].append(round(score / self.performance_count, 3))     # score/ rewards
                self.performance_before_inject[0].append(round(win_num / self.performance_count, 3))   # winning rate
                self.performance_before_inject[0].append(round(avg_diff / self.performance_count, 3))  # difference of the cash at the end of game
                self.logger.debug("Step # :" + str(num_test) + ' avg score : ' + str(round(score / num_test, 3)))
                self.logger.debug("Step # :" + str(num_test) + ' avg winning : ' + str(round(win_num / num_test, 3)))
                self.logger.debug("Step # :" + str(num_test) + ' avg diff : ' + str(round(avg_diff / num_test, 3)))


            # Check the novelty of game
            if env.output_kg_change() and self.novelty[-1] != env.output_kg_change():
                self.novelty.append(env.output_kg_change())
                self.logger.debug('Find the novelty in ' + str(num_test) +
                                  ' th game and will run novelty_state detection for every state now!')
                for novelty in env.output_kg_change():
                    self.novelty_spaces.add(novelty[0].replace('-', ' '))
        env.close()

    def find_novelty_state(self, interface, state):
        # TODO: add signal to retrain in this class.
        return interface.check_relative_state(state, self.novelty_spaces)



if __name__ == '__main__':
    hyp = Hyp_Learner()
    hyp.online_testing()
