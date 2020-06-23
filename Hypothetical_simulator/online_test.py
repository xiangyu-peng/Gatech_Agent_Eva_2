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
import shutil
from Hypothetical_simulator.logging_info import log_file_create
from monopoly_simulator_background.vanilla_A2C_main_v3 import MonopolyTrainer
from monopoly_simulator_background.evaluate_a2c import test_v2
import csv
logger = logging.getLogger('logging_info.online_testing')

class Config:
    device = torch.device('cuda:2')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default=upper_path + '/monopoly_simulator_background/weights/baseline.pkl', type=str)
    parser.add_argument('--device_name', default='cuda:2', type=str)
    parser.add_argument('--num_test', default=60000000, type=int)
    parser.add_argument('--performance_count', default=100, type=int)
    parser.add_argument('--retrain_type', default=None, type=str) # or hyp or baseline
    parser.add_argument('--config_file_offline_baseline', default='/Hypothetical_simulator/config_offline_baseline.ini', type=str)
    parser.add_argument('--config_file_offline_hyp', default='/Hypothetical_simulator/config_offline_hyp_1.ini', type=str)
    parser.add_argument('--config_file_online_baseline', default='/Hypothetical_simulator/config_online_baseline.ini', type=str)
    parser.add_argument('--config_file_online_hyp', default='/Hypothetical_simulator/config_online_hyp_1.ini', type=str)
    parser.add_argument('--config_file_online', default='/Hypothetical_simulator/config_online.ini', type=str)
    parser.add_argument('--upper_path', default='/media/becky/GNOME-p3', type=str)
    parser.add_argument('--result_csv_path', default='/Hypothetical_simulator/result_int_1.csv', type=str)
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
        self.interval = '_1'
        args = parse_args()
        self.device = torch.device(args['device_name'])
        self.model_path = args['model_path']
        self.model = torch.load(self.model_path)
        self.seed = args['seed']

        # game env/ novelty injection
        self.performance_before_inject = [[], [], []]
        self.novelty_spaces = set()
        self.novelty = [[]]
        self.upper_path = args['upper_path']
        self.config_file_online_baseline = args['config_file_online_baseline']
        self.config_file_online_hyp = args['config_file_online_hyp']
        self.config_file_online = args['config_file_online']

        #performance of agents
        self.num_test = args['num_test']
        self.performance_count = args['performance_count']  # default is 10; we average the performance of 10 games
        self.test_interval = 0

        # retraining parameters
        self.config_file_offline_baseline = args['config_file_offline_baseline']
        self.config_file_offline_hyp = args['config_file_offline_hyp']
        self.retrain_type = args['retrain_type']
        self.retrain_signal = False  # denotes if we will retrain the agent during next game
        self.trainer = self.set_offline_train_setting()

        self.retrain_steps = 0
        self.stop_sign = False

        # logger info
        if self.retrain_type:
            self.logger = log_file_create(
                upper_path + '/Hypothetical_simulator/log_testing_' + self.retrain_type + self.interval + '.log')
        else:
            self.logger = log_file_create(upper_path + '/Hypothetical_simulator/log_testing_noretrain.log')
        self.result_csv_path = self.upper_path + args['result_csv_path']  # save the retraining results to the csv


    def online_testing(self):
        """
        Online Testing
        1) Run the test with the pretrained model
        2) Detect the novelty
        3) Find the state involved with novelty
        4) Retrain the agent with the method of baseline and hypothetical way
        Baseline:
            Detect the novelty state and retrain from the scratch
        Hypothetical:
            Detect the novelty state and backward one step and retrain there.
        """
        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        if self.retrain_type == 'baseline':
            env.set_config_file(self.config_file_online_baseline) # half path
        elif self.retrain_type == 'hyp':
            env.set_config_file(self.config_file_online_hyp)  # half path
        else:
            env.set_config_file(self.config_file_online)

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
            save_signal = False
            novelty_state_num = []  # clear the novelty_related history

            while not done:
                round_num_game += 1
                s = s.reshape(1, -1)


                # check the novelty in the state
                # TODO: save all the game board before the novelty injection as well
                # TODO: Evaluation!
                # if self.novelty_spaces and not retrain_signal_per_game:
                #     # Find the state, involving novelty
                #     retrain_signal_per_game = self.find_novelty_state(env.output_interface(), s[0])

                # retrain the agent when the retrain_signal is triggered
                # if retrain_signal_per_game and not done_reset_bool and save_signal and self.retrain_type:
                #     new_model_path = self.offline_testing(retrain_type=self.retrain_type, num_test=num_test)
                #     #Update the agent
                #     self.model = torch.load(new_model_path)
                #     for i, (name, param) in enumerate(self.model.named_parameters()):
                #         if i == 0:
                #             print(i, (name, param))



                s = torch.tensor(s, device=self.device).float()
                prob = self.model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                if masked_actions[a[0]] == 0:  # check whether the action is valid
                    a = [1]
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])

                s = s_prime

                score_game += r
                # done_reset_bool = done


                # Make the game board used in retrain is the one step backward!
                # if self.novelty_spaces:
                #     if self.retrain_type == 'hyp':
                #         path_name = '/Hypothetical_simulator/game_board/gameboard_#' + str(
                #             num_test) + '_round_' + str(round_num_game) + '.json'
                #         env.save_gameboard(path_name)
                #     save_signal = True

                # save all the novelty related states for retraining
                if self.novelty_spaces and self.find_novelty_state(env.output_interface(), s):
                    novelty_state_num.append(round_num_game)
                    # save gameboard
                    path_name = '/Hypothetical_simulator/game_board/gameboard_#' + str(num_test) \
                                + '_round_' + str(round_num_game) + self.interval + '.json'
                    env.save_gameboard(path_name)

                    retrain_signal_per_game = True

            # done_reset_bool = True
            avg_diff += s[-2] - s[-1]
            score += score_game / round_num_game + 10 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0

            print(num_test)
            print(novelty_state_num)


            # retrain the agent when the retrain_signal is triggered
            if retrain_signal_per_game and self.retrain_type:
                if len(novelty_state_num) == 1 and novelty_state_num[0] == 1:
                    pass
                else:

                    new_model_path = self.offline_testing(retrain_type=self.retrain_type,
                                                           num_test=num_test,
                                                           novelty_set=novelty_state_num)
                    #Update the agent
                    self.model = torch.load(new_model_path)
                    self.model_path = new_model_path

            # Record the performance of the agent
            if num_test % self.performance_count == 0:
                self.performance_before_inject[0].append(round(score / self.performance_count, 3))     # score/ rewards
                self.performance_before_inject[0].append(round(win_num / self.performance_count, 3))   # winning rate
                self.performance_before_inject[0].append(round(avg_diff / self.performance_count, 3))  # difference of the cash at the end of game
                self.logger.debug("Step # :" + str(num_test) + ' avg score : ' + str(round(score / num_test, 3)))
                self.logger.debug("Step # :" + str(num_test) + ' avg winning : ' + str(round(win_num / num_test, 3)))
                self.logger.debug("Step # :" + str(num_test) + ' avg diff : ' + str(round(avg_diff / num_test, 3)))
                # refresh the count
                score = 0.0
                win_num = 0
                avg_diff = 0


            # Check the novelty of game
            if env.output_kg_change() and self.novelty[-1] != env.output_kg_change():
                self.novelty.append(env.output_kg_change())
                self.logger.debug('Find the novelty in ' + str(num_test) +
                                  ' th game and will run novelty_state detection for every state now!')
                self.logger.debug(str(env.output_kg_change()))
                for novelty in env.output_kg_change():
                    if novelty:
                        self.logger.debug('novelty is ' + novelty[0].replace('-', ' '))
                        self.novelty_spaces.add(novelty[0].replace('-', ' '))
        env.close()

    def find_novelty_state(self, interface, state):
        return interface.check_relative_state(state, self.novelty_spaces)

    def params_read(self, config_data, key_word):
        params = {}
        for key in config_data[key_word]:
            v = eval(config_data[key_word][key])
            params[key] = v
        return params

    def set_offline_train_setting(self):
        config_data = ConfigParser()
        config_data.read(self.upper_path + self.config_file_offline_hyp)
        params = self.params_read(config_data, 'hyper')

        # Begin retraining
        model_path = self.model_path
        # Set trainer:
        trainer = MonopolyTrainer(params=params,
                                  device_id=None,
                                  gameboard=None,
                                  kg_use=False,
                                  logger_use=False,
                                  config_file=self.config_file_offline_hyp,
                                  test_required=True,
                                  pretrain_model=model_path,
                                  seed=self.seed)
        return trainer

    def offline_testing(self, retrain_type='hyp', num_test=None, novelty_set=[]):
        """
        Off line training.
        Baseline: Change the env setting (novelty injection num) to 0
        # TODO: hypothetical testing

        :return : str, trained weight file path
        """
        # clear the folder or make a empty one
        save_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights' + \
                    '/' + retrain_type + '/' + str(num_test) + '/' + self.interval
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            os.makedirs(save_path)

        # Begin retraining
        for num in novelty_set:
            if num != 1:
                gameboard = '/Hypothetical_simulator/game_board/gameboard_#' + str(
                            num_test) +'_round_' + str(num) + self.interval + '.json'

                test_required = True if self.test_interval > 1000 else False
                if not self.stop_sign:
                    self.trainer.set_gameboard(gameboard=gameboard,
                                               seed=self.seed,
                                               save_path=save_path,
                                               test_required=test_required)
                else:
                    save_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights' + \
                                '/' + retrain_type + '/' + self.interval
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                        os.makedirs(save_path)
                    else:
                        os.makedirs(save_path)
                    self.trainer.set_gameboard(gameboard=None,
                                               seed=10,
                                               save_path=save_path,
                                               test_required=True,
                                               print_interval=1000,
                                               max_train_steps=500000,
                                               logger_use=True,
                                               logger_name=self.interval)


                loss_return, avg_score, avg_winrate, avg_diff = self.trainer.train()
                # print(loss_return, avg_score, avg_winrate, avg_diff)

                if avg_winrate >= 0.6:
                    self.stop_sign = True

                # model_path = save_path + '/hyp_v3_lr_0.0001_#_1.pkl'

                if test_required:
                    # avg_score, avg_winrate, avg_diff = self.trainer.test_v2(step_idx=self.retrain_steps,
                    #                                                         seed=0,
                    #                                                         config_file=self.config_file_offline_hyp,
                    #                                                         num_test=200)
                    # avg_score_2, avg_winrate_2, avg_diff_2 = test_v2(step_idx=self.retrain_steps,
                    #                                                  model=torch.load(save_path + '/hyp_v3_lr_0.0001_#_1.pkl'),
                    #                                                  device=self.device,
                    #                                                  num_test=200,
                    #                                                  seed=0,
                    #                                                  config_file=self.config_file_offline_hyp)

                    # self.logger.debug('Avg score is ' + str(avg_score))
                    self.logger.debug('Avg winning rate is ' + str(avg_winrate))
                    # self.logger.debug('Avg diff is ' + str(avg_diff))
                    self.logger.debug('=====================================')

                    # save the results to the cav file for figures
                    with open(self.result_csv_path, 'a', newline='') as csvfile:
                        resultwriter = csv.writer(csvfile, delimiter=' ',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        resultwriter.writerow([str(self.retrain_steps), str(avg_score), str(avg_winrate), str(avg_diff)])
                        # resultwriter.writerow([str(self.retrain_steps), str(avg_score_2), str(avg_winrate_2), str(avg_diff_2)])

                    self.test_interval = 0

                self.retrain_steps += 1/10
                self.test_interval += 1/10

        return save_path + '/hyp_v3_lr_0.0001_#_1.pkl'

if __name__ == '__main__':
    hyp = Hyp_Learner()
    hyp.online_testing()
