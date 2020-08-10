# nohup python online_test.py --novelty_change_num 18,0,18,0 --novelty_change_begin 1,0,1,0 --retrain_type gat --interval 1 --retrain_nums 10 --device_id 0 --seed 10 --novelty_introduce_begin 100,1000,2000,3000 --num_test 4000
import sys
import os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
sys.path.append(upper_path + '/GNN')
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
import datetime
from Hypothetical_simulator.logging_info import log_file_create
from monopoly_simulator_background.vanilla_A2C_main_v3 import MonopolyTrainer
from GNN.GAT import MonopolyTrainer_GAT
from monopoly_simulator_background.evaluate_a2c import test_v2
import csv
import copy
logger = logging.getLogger('logging_info.online_testing')
from monopoly_simulator_background import logger

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
    def __init__(self, args):
        # game env/ novelty injection
        # self.performance_before_inject = [[], [], []]
        self.novelty_spaces = set()
        self.novelty = [[]]
        self.novelty_newest = []
        self.upper_path = args.upper_path
        self.config_file = args.config_file
        # self.config_file_online_baseline = args['config_file_online_baseline']
        # self.config_file_online_hyp = args['config_file_online_hyp']
        # self.config_file_online = args['config_file_online']
        self.novelty_change_num = list(map(lambda x: int(x), args.novelty_change_num.split(',')))
        self.novelty_change_begin = list(map(lambda x: int(x), args.novelty_change_begin.split(',')))
        self.novelty_introduce_begin = list(map(lambda x: int(x), args.novelty_introduce_begin.split(',')))

        # exp_dict for env
        self.interval = '_' + str(args.interval)
        self.retrain_hyp_interval = int(args.interval)
        self.exp_dict_list, self.exp_dict_change_position = self.set_up_dict_list(self.novelty_change_num,
                                                                                  self.novelty_change_begin,
                                                                                  self.novelty_introduce_begin,
                                                                                  args.retrain_type)

        self.device = torch.device('cuda:' + args.device_id)
        self.device_id = args.device_id
        self.model_path = args.model_path
        self.model = torch.load(self.model_path, map_location={"cuda:1" : "cuda:" + self.device_id})
        self.seed = args.seed
        if args.retrain_type and 'baseline' not in args.retrain_type:
            self.adj = None
            self.adj_path = args.adj_path_folder + '_' + str(self.exp_dict_list[0]['novelty_num'][0]) + '_' + str(self.exp_dict_list[0]['novelty_num'][1])+ '_kg.npy'
            self.adj_path_new = args.adj_path_folder + '_' + str(self.exp_dict_list[0]['novelty_num'][0]) + '_' + str(self.exp_dict_list[0]['novelty_num'][1])+ '_kg.npy'

        #performance of agents
        self.num_test = args.num_test
        self.performance_count = args.performance_count  # default is 10; we average the performance of 10 games
        self.test_interval = 0  # ????

        # retraining parameters
        # self.config_file_offline_baseline = args['config_file_offline_baseline']
        # self.config_file_offline_hyp = args['config_file_offline_hyp']
        self.retrain_type = args.retrain_type
        self.retrain_signal = False  # denotes if we will retrain the agent during next game
        if args.retrain_type:
            self.generate_adj_first()
            self.trainer = self.set_offline_train_setting()
        self.retrain_steps = 0
        self.stop_sign = False
        self.retrain_nums = args.retrain_nums

        # logger info
        self.logger = log_file_create(self.upper_path + '/Hypothetical_simulator/logs/nov_' + \
            str(self.novelty_change_num) + '_' + str(self.novelty_change_begin) + \
            '_rt_' + str(self.retrain_nums)+ '_' + self.retrain_type + '_seed_' + str(self.seed) + '.log') \
            if self.retrain_type else \
            log_file_create(self.upper_path + '/Hypothetical_simulator/logs/nov_' + \
            str(self.novelty_change_num) + '_' + str(self.novelty_change_begin) + \
            '_wo_rt.log')

        if self.retrain_type:
            self.result_csv_str = '_seed_' + str(self.seed) + '_nov_' + str(self.novelty_change_num) +\
                                   '_' + str(self.novelty_change_begin) + '_rt_' + str(self.retrain_nums)+ \
                                   '_' + self.retrain_type  # save the retraining results to the csv
        else:
            self.result_csv_str = '_seed_' + str(self.seed) + '_nov_' + str(self.novelty_change_num) + \
                                  '_' + str(self.novelty_change_begin) + '_rt_' + str(self.retrain_nums)
        self.TB = logger.Logger(None, [logger.make_output_format('csv', 'log_test/', log_suffix=self.result_csv_str)])

    def set_up_dict_list(self, novelty_change_num_list, novelty_change_begin_list, novelty_introduce_begin_list, retrain_type):
        """
        Give you a list of dict to reset the game env novelty and a list of numbers of when to reset
        :param novelty_change_num_list:
        :param novelty_change_begin_list:
        :param novelty_introduce_begin_list:
        :param retrain_type:
        :return:
        """
        exp_dict_list = []
        exp_dict_change_position = []
        for i in range(len(novelty_change_num_list)):
            exp_dict = dict()
            exp_dict['novelty_num'] = (novelty_change_num_list[i], novelty_change_begin_list[i])
            exp_dict['novelty_inject_num'] = novelty_introduce_begin_list[i] if i == 0 else 0
            exp_dict['exp_type'] = 'kg' if retrain_type != 'baseline' else 'None'
            exp_dict_list.append(exp_dict)
            if i >= 1:
                exp_dict_change_position.append(novelty_introduce_begin_list[i])
        return exp_dict_list, exp_dict_change_position

    def load_adj(self):
        """
        load the relationship matrix
        :param path:
        :return:
        """
        if os.path.exists(self.adj_path):
            self.adj = np.load(self.adj_path)
            for i in range(self.adj.shape[1] // self.adj.shape[0] - 1):
                self.adj = self.adj[:, self.adj.shape[0] * i:self.adj.shape[0] * (i + 1)] + self.adj[:,
                                                                                            self.adj.shape[0] * (i + 1):
                                                                                            self.adj.shape[0] * (i + 2)]
            # print('Successful load game rule!')
        else:
            print(self.adj_path)
            raise FileNotFoundError

    def generate_adj_first(self):
        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        env.set_config_file(self.config_file)
        env.set_exp(self.exp_dict_list[0])
        env.set_kg(True)  # we need to use knowledge graph to detect the game rule change
        env.set_board()  # we use the original game-board in the beginning
        env.seed(self.seed)
        with HiddenPrints():
            env.reset()
        for _ in range(10):
            done = False
            while not done:
                with HiddenPrints():
                    _, _, done, _ = env.step(1)
        print('kg generate')

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
        env.set_config_file(self.config_file)
        env.set_exp(self.exp_dict_list[0])
        env.set_kg(True)  # we need to use knowledge graph to detect the game rule change
        env.set_board()  # we use the original game-board in the beginning
        env.seed(self.seed)

        # prepare for reset game env
        env_set_num = 0  # use which exp_dict to reset the game env
        env_stop_change_bool = True if len(self.exp_dict_change_position) == 0 else False  # indicates whether to reset the game env
        exp_dict_change_position = self.exp_dict_change_position  # we will pop the used ones, so we make a copy

        # Begin test
        retrained_bool = False
        score = 0.0
        done = False
        win_num = 0
        avg_diff = 0

        with HiddenPrints():  # reset the env
            s, masked_actions = env.reset()

        # num_test = 0  # count the # of games before novelty injection
        done_reset_bool = False
        retrain_signal_per_game = False  # define if we need to retrain tall the states after this state
        novelty_state_num = []  # clear the novelty_related history

        for num_test in range(1, self.num_test):
            round_num_game, score_game = 0, 0

            save_signal = False
            # retrain the agent when the retrain_signal is triggered
            if retrain_signal_per_game and self.retrain_type:
                if done:
                    with HiddenPrints():
                        s, r, done, masked_actions = env.step(1)
                if 'hyp' not in self.retrain_type:
                    path_name = '/Hypothetical_simulator/game_board/gameboard_' + self.result_csv_str + '.json'
                    env.save_gameboard(path_name)
                else:
                    path_name = None
                self.offline_testing(num_test=num_test,
                                     novelty_set=novelty_state_num,
                                     retrained_bool=retrained_bool,
                                     game_board=path_name,
                                     )
                # Update the agent
                self.model = self.trainer.model
                # self.model_path = new_model_path
                retrained_bool = True
                if args.retrain_type and 'baseline' not in args.retrain_type:
                    self.load_adj()
                novelty_state_num = []  # clear the novelty_related history

            done = False
            start_time = datetime.datetime.now()
            while not done:
                round_num_game += 1
                s = s.reshape(1, -1)
                s = torch.tensor(s, device=self.device).float()

                if args.retrain_type and 'baseline' not in args.retrain_type and retrained_bool:
                    s = self.model.forward(s, self.adj)

                # check the novelty in the state
                # TODO: save all the game board before the novelty injection as well
                # # TODO: Evaluation!
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
                #         path_name = '/Hypothetical_simulator/game_board/gameboard_round_' + str(round_num_game) + '.json'
                #         env.save_gameboard(path_name)

                # save all the novelty related states for retraining
                if self.novelty_spaces and self.find_novelty_state(env.output_interface(), s):
                    novelty_state_num.append(round_num_game)
                    # save gameboard
                    if self.retrain_type and 'hyp' in self.retrain_type:
                        path_name = '/Hypothetical_simulator/game_board/gameboard_round_' + str(round_num_game) + \
                                    self.interval + self.result_csv_str + '.json'
                        env.save_gameboard(path_name)
                    retrain_signal_per_game = True

            # done_reset_bool = True
            avg_diff += 0
            score += score_game / round_num_game + 10 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)

            # Record the performance of the agent
            if num_test % self.performance_count == 0:
                # self.performance_before_inject[0].append(round(score / self.performance_count, 3))     # score/ rewards
                # self.performance_before_inject[0].append(round(win_num / self.performance_count, 3))   # winning rate
                # self.performance_before_inject[0].append(round(avg_diff / self.performance_count, 3))  # difference of the cash at the end of game
                # self.logger.debug("Step # :" + str(num_test) + ' avg score : ' + str(round(score / num_test, 3)))
                self.logger.debug("Step # :" + str(num_test) + ' avg winning : ' + str(round(win_num / self.performance_count, 3)))
                # self.logger.debug("Step # :" + str(num_test) + ' avg diff : ' + str(round(avg_diff / num_test, 3)))
                self.TB.logkv('step_idx', num_test)
                self.TB.logkv('avg_winning', round(win_num / self.performance_count, 3))
                if 'hyp' in self.retrain_type:
                    self.TB.logkv('retrain_steps', self.retrain_steps)
                self.TB.dumpkvs()
                print(num_test, round(win_num / self.performance_count, 3),'retrain_step', self.retrain_steps)

                # refresh the count
                score = 0.0
                win_num = 0
                avg_diff = 0

            # Check the novelty of game
            env_kg = copy.deepcopy(env.output_kg_change())

            if env_kg and self.novelty[-1] != env_kg:
                new_novelty = []
                for novelty in env_kg:
                    add_bool = True
                    for novelty_history in self.novelty:
                        if novelty in novelty_history:
                            add_bool = False
                    if add_bool:
                        new_novelty.append(novelty)

                self.novelty.append(env_kg)
                self.novelty_newest = new_novelty

                self.logger.debug('Find the novelty in ' + str(num_test) +
                                  ' th game and will run novelty_state detection for every state now!')
                self.logger.debug(str(env.output_kg_change()))

                self.novelty_spaces.clear()
                for novelty in self.novelty_newest:
                    if novelty:
                        self.logger.debug('novelty is ' + novelty[0].replace('-', ' '))
                        self.novelty_spaces.add(novelty[0].replace('-', ' '))
                print('self.novelty_spaces', self.novelty_spaces)

                print('New novelty found in ', num_test, ' th test.')
                # print('self.novelty', self.novelty[-1])
                print('self.novelty_newest', self.novelty_newest)

                if self.retrain_type and 'hyp' not in self.retrain_type:
                    retrain_signal_per_game = True
                if self.retrain_type and 'baseline' not in self.retrain_type:
                    self.adj_path = self.adj_path_new


            # reset the game env
            if not env_stop_change_bool and num_test == int(self.exp_dict_change_position[0]):
                env_set_num += 1
                env.set_exp(self.exp_dict_list[env_set_num])
                env.seed(self.seed + env_set_num)
                exp_dict_change_position.pop(0)
                env_stop_change_bool = True if len(exp_dict_change_position) == 0 else False
                if self.retrain_type and 'baseline' not in self.retrain_type:
                    self.adj_path_new = args.adj_path_folder + '_' + str(self.exp_dict_list[env_set_num]['novelty_num'][0]) + \
                                        '_' + str(self.exp_dict_list[env_set_num]['novelty_num'][1]) + '_kg.npy'
                print(num_test, self.exp_dict_list[env_set_num])

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
        trainer = None
        exp_dict = self.exp_dict_list[0].copy()
        exp_dict['novelty_inject_num'] = sys.maxsize
        # Begin retraining
        model_path = self.model_path
        config_data = ConfigParser()
        config_data.read(self.upper_path + self.config_file)
        params = self.params_read(config_data, 'hyper')

        if self.retrain_type == 'baseline':
            # Set trainer:
            trainer = MonopolyTrainer(params=params,
                                      device_id=self.device_id,
                                      gameboard=None,
                                      kg_use=False,
                                      logger_use=False,
                                      config_file=args.config_file,
                                      test_required=False,
                                      pretrain_model=None,
                                      tf_use=True,
                                      tf_stop_num=20000,
                                      seed=self.seed + 1,
                                      exp_dict=exp_dict)
        else:
            # Set trainer:
            trainer = MonopolyTrainer_GAT(params,
                                          device_id=self.device_id,
                                          gameboard=None,
                                          kg_use=False,
                                          logger_use=False,
                                          config_file=args.config_file,
                                          test_required=False,
                                          tf_use=False,
                                          pretrain_model=None,
                                          gat_use='kg',
                                          seed=args.seed,
                                          exp_dict=exp_dict,
                                          tf_stop_num=20000)

        return trainer

    def offline_testing(self, num_test=None, novelty_set=[], retrained_bool=False, game_board=None):
        """
        Off line training.
        Baseline: Change the env setting (novelty injection num) to 0
        # TODO: hypothetical testing

        :return : str, trained weight file path
        """
        #clear the folder or make a empty one
        # save_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights' + \
        #             '/' + self.retrain_type + '/' + str(num_test)
        save_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights' + \
                    '/' + self.retrain_type + '/' + self.result_csv_str
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            os.makedirs(save_path)

        # save_name = None

        # Set up the trainer
        save_name = None
        if 'hyp' not in self.retrain_type:
            self.trainer.set_gameboard(gameboard=game_board,
                                       save_path=save_path,
                                       print_interval=self.retrain_nums,
                                       max_train_steps=self.retrain_nums + 1)
            if 'baseline' not in self.retrain_type:
                self.trainer.set_gameboard(adj_path=self.adj_path)

            save_name = self.trainer.train()

        else:
        # Begin retraining
            for num in novelty_set:
                gameboard = '/Hypothetical_simulator/game_board/gameboard_round_' + str(num) + self.interval + '.json'
                # if not self.stop_sign:
                self.trainer.set_gameboard(gameboard=game_board,
                                           save_path=save_path,
                                           print_interval=self.retrain_hyp_interval,
                                           max_train_steps=self.retrain_hyp_interval + 1,
                                           adj_path=self.adj_path)
                save_name = self.trainer.train()

            self.retrain_steps += self.retrain_hyp_interval * len(novelty_set)
                # else:
                #     save_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights' + \
                #                 '/' + retrain_type + '/' + self.interval
                #     if os.path.exists(save_path):
                #         shutil.rmtree(save_path)
                #         os.makedirs(save_path)
                #     else:
                #         os.makedirs(save_path)
                #     self.trainer.set_gameboard(gameboard=None,
                #                                seed=10,
                #                                save_path=save_path,
                #                                test_required=True,
                #                                print_interval=1000,
                #                                max_train_steps=500000,
                #                                logger_use=True,
                #                                logger_name=self.interval)


                    # loss_return, avg_score, avg_winrate, avg_diff = self.trainer.train()
                    # # print(loss_return, avg_score, avg_winrate, avg_diff)
                    #
                    # if avg_winrate >= 0.6:
                    #     self.stop_sign = True
                    #
                    # # model_path = save_path + '/hyp_v3_lr_0.0001_#_1.pkl'
                    #
                    # if test_required:
                    #     # avg_score, avg_winrate, avg_diff = self.trainer.test_v2(step_idx=self.retrain_steps,
                    #     #                                                         seed=0,
                    #     #                                                         config_file=self.config_file_offline_hyp,
                    #     #                                                         num_test=200)
                    #     # avg_score_2, avg_winrate_2, avg_diff_2 = test_v2(step_idx=self.retrain_steps,
                    #     #                                                  model=torch.load(save_path + '/hyp_v3_lr_0.0001_#_1.pkl'),
                    #     #                                                  device=self.device,
                    #     #                                                  num_test=200,
                    #     #                                                  seed=0,
                    #     #                                                  config_file=self.config_file_offline_hyp)
                    #
                    #     # self.logger.debug('Avg score is ' + str(avg_score))
                    #     self.logger.debug('Avg winning rate is ' + str(avg_winrate))
                    #     # self.logger.debug('Avg diff is ' + str(avg_diff))
                    #     self.logger.debug('=====================================')
                    #
                    #     # save the results to the cav file for figures
                    #     with open(self.result_csv_path, 'a', newline='') as csvfile:
                    #         resultwriter = csv.writer(csvfile, delimiter=' ',
                    #                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #         resultwriter.writerow([str(self.retrain_steps), str(avg_score), str(avg_winrate), str(avg_diff)])
                    #         # resultwriter.writerow([str(self.retrain_steps), str(avg_score_2), str(avg_winrate_2), str(avg_diff_2)])
                    #
                    #     self.test_interval = 0
                    #
                    # self.retrain_steps += 1/10
                    # self.test_interval += 1/10


        return save_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file',
                        default='/monopoly_simulator_background/config.ini',
                        required=False,
                        help="config_file for env")
    parser.add_argument('--novelty_change_num',
                        type=str,
                        default='0',
                        required=True,
                        help="Novelty price change number, use 5,6,7 to indicate 3 novelty change num")
    parser.add_argument('--novelty_change_begin',
                        type=str,
                        default='0',
                        required=True,
                        help="Novelty price change begin number")
    parser.add_argument('--novelty_introduce_begin',
                        type=str,
                        default='100',
                        required=True,
                        help="Novelty price change begin number, use 100, 200, 300 to indicate the novelty change place")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        required=True,
                        help="Env seed")
    parser.add_argument('--upper_path',
                        type=str,
                        default='/media/becky/GNOME-p3',
                        required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--exp_type',
                        type=str,
                        default='state',
                        required=False,
                        help="Use state kg or not. Choose from state and kg")
    parser.add_argument('--device_id',
                        default='1',
                        type=str,
                        required=False,
                        help="GPU id we use")
    parser.add_argument('--model_path',
                        default='/media/becky/GNOME-p3/monopoly_simulator_background/weights/del_ran_19_1_v3_lr_0.001_#_100.pkl',
                        type=str,
                        required=False,
                        help="Pre-trained model")
    parser.add_argument('--num_test',
                        default=50000,
                        type=int,
                        required=False,
                        help="# of tests")
    parser.add_argument('--performance_count',
                        default=50,
                        type=int,
                        required=False,
                        help="???")
    parser.add_argument('--retrain_type',
                        default=None,
                        required=False,
                        help="baseline or hyp")
    parser.add_argument('--result_csv_path',
                        default='/Hypothetical_simulator/',
                        type=str,
                        required=False,
                        help="result of online test -> csv file")
    parser.add_argument('--interval',
                        default=1,
                        type=int,
                        required=True,
                        help="interval?")
    parser.add_argument('--retrain_nums',
                        default=10,
                        type=int,
                        required=True,
                        help="For every retrain, how many steps we will use")
    parser.add_argument('--adj_path_folder',
                        default='/media/becky/GNOME-p3/KG_rule/matrix_rule/kg_matrix',
                        type=str,
                        required=False,
                        help="The folder has the npz matrix file for adj of game rule/ kg")
    args = parser.parse_args()

    hyp = Hyp_Learner(args)
    hyp.online_testing()
