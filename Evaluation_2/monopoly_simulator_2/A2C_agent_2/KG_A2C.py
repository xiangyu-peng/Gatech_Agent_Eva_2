# With knowledge graph development
# Only consider 2 actions
# Only take one action each time
# log name + save name
# add kg gat into the model
# nohup python GAT_part.py --pretrain_model /media/becky/GNOME-p3/monopoly_simulator_background/weights0_0_gat_part_seed_1048000.pkl --device_id 2 --novelty_change_num 18 --novelty_change_begin 1 --novelty_introduce_begin 0 --retrain_type gat_part --exp_name 18_1_v --seed 10

import numpy as np
# from GNN.model import *
# adj = np.load('/media/becky/GNOME-p3/KG_rule/kg_matrix.npy')
import sys, os
upper_path = os.path.abspath('.').replace('/Evaluation_2/monopoly_simulator_2','')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path + '/GNN')
####################
from monopoly_simulator_background.vanilla_A2C import *
from monopoly_simulator_background.interface import Interface
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent

from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from monopoly_simulator_background.gameplay_tf import *
from monopoly_simulator_background import logger
from itertools import product
import argparse
import warnings
warnings.filterwarnings('ignore')
import datetime
import copy
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MonopolyTrainer_GAT:
    def __init__(self,
                 params,
                 device_id=0,
                 gameboard=None,
                 kg_use=True,
                 logger_use=True,
                 config_file=None,
                 test_required=True,
                 tf_use=True,
                 pretrain_model=None,
                 tf_stop_num=0,
                 exp_name=None,
                 retrain_type='baseline',
                 exp_dict=None,
                 adj_path=None,
                 seed=0,
                 len_vocab=92):  # unique exp name with number!

        # set up the novelty and the file name/ matrix path
        self.exp_name = exp_name
        self.retrain_type = retrain_type
        self.novelty = [[]]
        self.novelty_newest = []
        self.exp_dict = exp_dict
        # self.novelty_change_num = list(map(lambda x: int(x), args.novelty_change_num.split(',')))
        # self.novelty_change_begin = list(map(lambda x: int(x), args.novelty_change_begin.split(',')))
        # self.novelty_introduce_begin = list(map(lambda x: int(x), args.novelty_introduce_begin.split(',')))
        # self.exp_dict_list, self.exp_dict_change_position = self.set_up_dict_list(self.novelty_change_num,
        #                                                                           self.novelty_change_begin,
        #                                                                           self.novelty_introduce_begin,
        #                                                                           retrain_type=exp_name+'_'+self.retrain_type+'_seed_'+ str(args.seed))
        # self.exp_dict = self.exp_dict_list[0]
        self.adj_path = None
        if 'gat' in retrain_type:
            self.adj = None
            self.adj_path = adj_path
            print('self.adj_path', self.adj_path)

        self.params = params
        self.input_vocab_size = self.params['input_vocab_size']
        self.gat_output_size = self.params['gat_output_size']
        self.seed = seed
        self.tf_use = tf_use
        self.tf_stop_num = tf_stop_num
        self.config_file = config_file
        self._device_id = 'cuda:' + device_id if device_id != '-1' else 'cpu'

        self.PRINT_INTERVAL = self.params['print_interval']
        self.n_train_processes = self.params['n_train_processes']  # batch size
        self.learning_rate = self.params['learning_rate']
        self.update_interval = self.params['update_interval']
        self.gamma = self.params['gamma']
        self.max_train_steps = self.params['max_train_steps']
        self.hidden_state = self.params['hidden_state']
        self.action_space = self.params['action_space']
        self.state_num = self.params['state_num']
        self.actor_loss_coefficient = self.params['actor_loss_coefficient']
        self.kg_vector = []
        self.interface = Interface()
        self.save_path = self.params['save_path']
        self.num_test = self.params['num_test']
        self.gameboard = gameboard
        self.kg_use = kg_use
        self.logger_use = logger_use
        self.test_result = {'step': [], 'loss': [],
                     'winning_rate': [], 'avg_score': [], 'avg_diff': [], 'avg_score_no': []}
        self.spend_time = 0
        self.test_required = test_required


        # config logger
        if self.logger_use:
            hyper_config_str = '_n' + str(self.n_train_processes) + '_lr' + str(self.learning_rate) + '_ui' + str(
                self.update_interval) \
                               + '_y' + str(self.gamma) + '_s' + str(self.max_train_steps) + '_hs' + str(self.hidden_state) \
                               + '_as' + str(self.action_space) + '_sn'  \
                               + '_ac' + str(self.actor_loss_coefficient) + '_seed' + str(self.seed) + '_novelty_' \
                               + exp_name + '_' + self.retrain_type
            self.TB = logger.Logger(None, [logger.make_output_format('csv', 'logs/', log_suffix=hyper_config_str)])


        ###set to cpu for evaluation
        self.device = torch.device(self._device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

        # Set the env for training
        with HiddenPrints():
            self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use, '/Evaluation_2/monopoly_simulator_2/A2C_agent_2/' + self.config_file, self.seed, self.exp_dict)

        if self.gameboard:
            self.interface.set_board(self.gameboard)
        else:
            self.interface.set_board('/monopoly_simulator_background/baseline_interface.json')

        # generate kg
        # if not kg_use:
        #     self.generate_kg()

        if pretrain_model:
            self.model = torch.load(pretrain_model, map_location={"cuda:2" : self._device_id})
        else:
            if 'gat' in self.retrain_type:
                self.config_model = ConfigParser()
                self.config_model.hidden_state = self.params['hidden_state']
                self.config_model.action_space = self.params['action_space']
                self.config_model.state_num = self.params['state_num']
                self.config_model.gat_output_size = self.params['gat_output_size']
                self.config_model.gat_emb_size = self.params['gat_emb_size']
                self.config_model.embedding_size = self.params['embedding_size']
                self.config_model.dropout_ratio = self.params['dropout_ratio']
                self.config_model.state_output_size = self.params['state_output_size']
                self.config_model.len_vocab = len_vocab
                print('self.config_model.len_vocab', self.config_model.len_vocab, len_vocab)

                if self.gameboard:
                    self.config_model.state_num = len(self.interface.board_to_state(self.gameboard))

                self.model = ActorCritic(self.config_model, device=self.device, gat_use='kg')  # A2C model
            else:
                self.config_model = ConfigParser()
                self.config_model.hidden_state = self.params['hidden_state']
                self.config_model.action_space = self.params['action_space']
                self.config_model.state_num = self.params['state_num']
                self.config_model.state_output_size = self.params['state_output_size']

                if self.gameboard:
                    self.config_model.state_num = len(self.interface.board_to_state(self.gameboard))
                print('self.config_model.state_num', self.config_model.state_num)
                self.model = ActorCritic(self.config_model, gat_use=False, device=self.device)  # A2C model
        self.len_vocab = len_vocab
        self.model.to(self.device)
        self.loss = 0
        import torch.nn as nn
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # for p in self.model.parameters():
        #     if p.requires_grad:
        #         print(p.name)
        self.memory = Memory()

        self.novelty_spaces = set()
        for (i, name) in enumerate(
                ["Mediterranean Avenue", "Baltic Avenue", "Reading Railroad", "Oriental Avenue", "Vermont Avenue",
                 "Connecticut Avenue", "St. Charles Place", "Electric Company"]):
            if i + 1 >= 7 and i + 1 <= 7:
                self.novelty_spaces.add(name)

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
        print('novelty_change_num_list', novelty_change_num_list)
        for i in range(len(novelty_change_num_list)):
            exp_dict = dict()
            exp_dict['novelty_num'] = (novelty_change_num_list[i], novelty_change_begin_list[i])
            exp_dict['novelty_inject_num'] = 0
            exp_dict['exp_type'] = retrain_type
            exp_dict_list.append(exp_dict)
            if i >= 1:
                exp_dict_change_position.append(novelty_introduce_begin_list[i] * 1000)
        print(exp_dict_list, exp_dict_change_position)
        return exp_dict_list, exp_dict_change_position

    def reinitialize_agent(self, gameboard=None):
        # if gameboard:
        #     self.config_model.state_num = len(self.interface.board_to_state(gameboard))
        self.model = ActorCritic(self.config_model, device=self.device, gat_use='kg')  # A2C model
        self.model.to(self.device)

    def add_vector_to_state(self, states, vector, device):
        state_return = []
        for state in states:
            new_state = np.concatenate((vector.reshape(1, -1), [state]), axis=1)
            state_return.append(new_state[0])
        return torch.tensor(state_return, device=device).float()

    def generate_kg(self):
        if os.path.exists(self.adj_path):
            print('File exist!')
            return True
        env = gym.make('monopoly_simple-v1')
        env.set_config_file('/Evaluation_2/monopoly_simulator_2/A2C_agent_2/' + self.config_file)
        env.set_exp(self.exp_dict)
        env.set_kg(True)
        env.set_board()  # self.gameboard
        env.seed(seed=0)
        with HiddenPrints():
            env.reset()
        for _ in range(10):
            done = False
            while not done:
                with HiddenPrints():
                    _, _, done, _ = env.step(1)
        print('kg generate')
        return True

    def load_adj(self):
        """
        load the relationship matrix
        :param path:
        :return:
        """
        if os.path.exists(self.adj_path):
            print('adj_use in KG-a2c is ',self.adj_path)
            self.adj = np.load(self.adj_path)
            adj_return = np.zeros((self.adj.shape[0], self.adj.shape[0]))
            print('adj_return', adj_return)
            for i in range(self.adj.shape[1] // self.adj.shape[0]):
                adj_return += self.adj[:,self.adj.shape[0] * i:self.adj.shape[0] * (i+1)]
            self.adj = adj_return

        else:
            self.adj = np.zeros((self.len_vocab, self.len_vocab))

    def test_v2(self, step_idx, seed, config_file=None, num_test=200):
        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        env.set_config_file(config_file)
        env.set_exp(self.exp_dict)
        env.set_kg(False)
        env.set_board(self.gameboard) # self.gameboard
        env.seed(seed)
        score = 0.0
        done = False
        win_num = 0
        avg_diff = 0
        score_no = 0

        with HiddenPrints():
            s, masked_actions = env.reset()

        for _ in range(num_test):
            num_game = 0
            score_game = 0
            score_no_game = 0
            done_type = True
            while not done:
                num_game += 1
                s = s.reshape(1, -1)
                s = torch.tensor(s, device=self.device).float()
                if 'gat' in self.retrain_type:
                    s = self.model.forward(s, self.adj)
                else:
                    s = self.model.forward_baseline(s)
                prob = self.model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                if masked_actions[a[0]] == 0:
                    a = [1]
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])
                s = s_prime

                if done_type:
                    done_type = False

                score_game += r
                score_no_game += r

            avg_diff += s[-2] - s[-1]
            score += score_game/num_game  + 1 * abs(abs(int(done) - 2) - 1)
            score_no += score_no_game/num_game
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0

        if self.logger_use:
            print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
            print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
            print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")

            end_time = datetime.datetime.now()
            self.spend_time = (end_time - self.start_time).seconds / 60 / 60  # hr

        env.close()

        return round(score/num_test, 5), round(win_num/num_test, 5), round(avg_diff/num_test, 5), round(score_no/num_test, 5)

    def save(self, step_idx):
        save_name = self.save_path + str(step_idx // self.PRINT_INTERVAL) + '.pkl'
        # torch.save(self.model.state_dict(), save_name)
        torch.save(self.model, save_name)
        print('save to =>', save_name)
        return save_name

    def check_novelty(self):
        # Check the novelty of game
        env_kg = self.envs.output_novelty()
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
            print('Find the novelty')
            self.novelty_list = []
            for i in self.novelty_newest:
                self.novelty_list.append(self.novelty_newest[0])
            print('self.novelty_newest', self.novelty_list)

    def set_gameboard(self,
                      gameboard=None,
                      save_path=None,
                      print_interval=None,
                      max_train_steps=None,
                      pretrain_model=None,
                      exp_dict=None,
                      adj_path=None,
                      seed=None,
                      learning_rate=None):

        self.learning_rate = learning_rate if learning_rate else self.learning_rate
        self.seed = seed if seed else self.seed
        self.gameboard = gameboard
        with HiddenPrints():
            self.envs.close()
            self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use, '/Evaluation_2/monopoly_simulator_2/A2C_agent_2/' + self.config_file,
                                    self.seed,
                                    self.exp_dict)
        self.save_path = save_path if save_path else self.save_path
        # self.seed = seed if seed else self.seed
        # self.test_required = test_required
        self.PRINT_INTERVAL = print_interval if print_interval else self.PRINT_INTERVAL
        self.max_train_steps = max_train_steps if max_train_steps else self.max_train_steps
        # if logger_use == True:
        #     self.logger_use = True
        #     hyper_config_str = '_n' + str(self.n_train_processes) + '_lr' + str(self.learning_rate) + '_ui' + str(
        #         self.update_interval) \
        #                        + '_y' + str(self.gamma) + '_s' + str(self.max_train_steps) + '_hs' + str(self.hidden_state) \
        #                        + '_as' + str(self.action_space) + '_sn'  \
        #                        + '_ac' + str(self.actor_loss_coefficient) + '_hyp_' + logger_name
        #
        #     self.TB = logger.Logger(None, [logger.make_output_format('csv', 'logs/', log_suffix=hyper_config_str)])
        if pretrain_model:
            # self.model = torch.load(pretrain_model, map_location={"cuda:2": "cuda:1"})
            self.model.load_state_dict(torch.load(pretrain_model))
        if exp_dict:
            self.exp_dict = exp_dict
            with HiddenPrints():
                self.envs.close()
                self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use, '/Evaluation_2/monopoly_simulator_2/A2C_agent_2/' + self.config_file,
                                        self.seed,
                                        self.exp_dict)
        self.adj_path = adj_path if adj_path else self.adj_path

    def train(self):
        # prepare for reset game env
        # env_set_num = 0  # use which exp_dict to reset the game env
        # env_stop_change_bool = True if len(self.exp_dict_change_position) == 0 else False  # indicates whether to reset the game env
        # exp_dict_change_position = self.exp_dict_change_position  # we will pop the used ones, so we make a copy

        self.start_time = datetime.datetime.now()
        self.spend_time = 0

        if 'winning_rate' in self.test_result.keys():
            print('keep training', self.test_result['winning_rate'])
        else:
            print('begin retrain!')
        step_idx = 1
        save_name = None
        converge_signal = False
        with HiddenPrints():
            reset_array = self.envs.reset()

        s, masked_actions, background_actions = [reset_array[i][0] for i in range(len(reset_array))], \
                            [reset_array[i][1][0] for i in range(len(reset_array))],\
                            [reset_array[i][1][1] for i in range(len(reset_array))]
        loss_train = torch.tensor(0, device=self.device).float()
        while step_idx < self.max_train_steps and self.spend_time < 0.1:  # TODO change to 2.5 hr
            loss = torch.tensor(0, device=self.device).float()
            for _ in range(self.update_interval):
                entropy = 0
                log_probs, masks, rewards, values = [], [], [], []
                s = torch.tensor(s, device=self.device).float()
                if 'gat' in self.retrain_type:
                    self.load_adj()
                    s = self.model.forward(s, self.adj)
                else:
                    s = self.model.forward_baseline(s)

                prob = self.model.actor(s)  # s => tensor #output = prob for actions

                a = []
                for i in range(self.n_train_processes):
                    a_once = Categorical(prob).sample().cpu().numpy()[i]  # substitute
                    if masked_actions[i][a_once] == 0:
                        a_once = 1
                    a.append(a_once)
                # while opposite happens. Step won't change env, step_nochange changes the env\
                # if self.tf_use:
                s_prime_cal, r, done, info = self.envs.step_nochange(a)

                values.append(self.model.critic(s))

                log_prob = Categorical(prob).log_prob(torch.tensor(a, device=self.device))
                entropy += Categorical(prob).entropy().mean()
                log_probs.append(log_prob)
                rewards.append(torch.FloatTensor(r).unsqueeze(1).to(self.device))

                done = [[0] if i > 0 else [1] for i in done]
                masks.append(torch.tensor(done, device=self.device).float())

                # Teacher forcing part #
                # a_tf = []
                # for i in range(self.n_train_processes):
                #     if masked_actions[i][0] == 1:
                #         a_tf.append(0)
                #     else:
                #         a_tf.append(1)
                if self.tf_use:
                    # a_tf = [1 for i in range(self.n_train_processes)]
                    if step_idx < self.tf_stop_num: #background_actions
                        s_prime, _, _, info = self.envs.step_after_nochange(background_actions)  # background_actions
                    else:
                        s_prime, _, _, info = self.envs.step_after_nochange(a)

                else:
                    # s = s_prime_cal
                    s_prime, _, _, info = self.envs.step_after_nochange(a)

                s = s_prime
                #########################
                masked_actions = [info_[0] for info_ in info]  # Update masked actions
                background_actions = [info_[1] for info_ in info]

                ##########
                s_prime_cal = torch.tensor(s_prime_cal, device=self.device).float()

                if 'gat' in self.retrain_type:
                    self.load_adj()
                    s_prime_cal = self.model.forward(s_prime_cal, self.adj)
                else:
                    s_prime_cal = self.model.forward_baseline(s_prime_cal)


                # loss cal
                log_probs = torch.cat(log_probs)
                returns = compute_returns(self.model.critic(s_prime_cal), rewards, masks, gamma=0.99)
                # print('rewards', rewards)
                # print(self.model.critic(s_prime_cal))

                returns = torch.cat(returns).detach()
                # print('returns',returns)
                values = torch.cat(values)
                advantage = returns - values
                # print('Advan',rewards, advantage, a, background_actions, s)
                # print('log_probs',log_probs)
                actor_loss = -(log_probs * advantage.detach()).mean()
                # print('actor_loss',actor_loss)
                critic_loss = advantage.pow(2).mean()
                # print('critic_loss',critic_loss)
                loss_once = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                if not math.isnan(loss_once.detach().numpy()):
                    loss += loss_once
                self.memory.clear()
                # print('loss', loss)

            loss /= self.update_interval
            self.loss = loss
            loss_train += loss

            if loss != torch.tensor(0, device=self.device) and loss != torch.tensor(float('nan'), device=self.device):
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                # print(self.optimizer.step())
            # print(torch.norm(self.model.graph_gat.gat.attentions[0].W))
            # print(torch.norm(self.model.fc_1.weight))
            avg_score, avg_winning, avg_diff, avg_score_no = 0, 0, 0, 0
            # self.check_novelty()
            if step_idx % self.PRINT_INTERVAL == 0:
                # self.check_novelty()
                # save weights of A2C
                save_name = self.save(step_idx)

                loss_return = loss_train.cpu().detach().numpy()
                loss_train = torch.tensor(0, device=self.device).float()

                if self.test_required:
                    avg_score, avg_winning, avg_diff, avg_score_no = self.test_v2(step_idx, seed=0, config_file='/Evaluation_2/monopoly_simulator_2/A2C_agent_2/' + self.config_file)
                    print(avg_score, avg_winning, avg_diff, avg_score_no)
                    # Add test result to storage and then plot
                    self.test_result['step'].append(step_idx // self.PRINT_INTERVAL)
                    self.test_result['loss'].append(round(float(loss_return)) / self.PRINT_INTERVAL)
                    self.test_result['winning_rate'].append(avg_winning)
                    self.test_result['avg_score'].append(avg_score)
                    self.test_result['avg_diff'].append(avg_diff)
                    self.test_result['avg_score_no'].append(avg_score_no)

                if self.logger_use:
                    self.TB.logkv('step_idx', step_idx)
                    self.TB.logkv('loss_train', round(float(loss_return) / self.PRINT_INTERVAL, 3))
                    if self.test_required:
                        self.TB.logkv('avg_score', avg_score)
                        self.TB.logkv('avg_winning', avg_winning)
                        self.TB.logkv('avg_diff', avg_diff)
                        self.TB.logkv('avg_score_no', avg_score_no)
                    self.TB.dumpkvs()

                    # plot the results
                    # print(self.model.named_parameters())
                    # for i, (name, param) in enumerate(self.model.named_parameters()):
                    #     if 'bn' not in name:
                    #         self.writer.add_histogram(name, param, 0)
                    #         self.writer.add_scalar('loss', loss_train / self.PRINT_INTERVAL, i)
            step_idx += 1
            end_time = datetime.datetime.now()

            self.spend_time = (end_time - self.start_time).seconds / 60 / 60

            # After converge, we will stop the training TODO change the number here
            if self.test_required:
                if len(self.test_result['winning_rate']) > 50:
                    if max(self.test_result['winning_rate'][:-20]) > max(self.test_result['winning_rate'][-20:]) + 0.05:
                        converge_signal = True
                        break
                    if (max(self.test_result['winning_rate'][-15:]) - min(self.test_result['winning_rate'][-15:])) < 0.03:
                        converge_signal = True
                        break

            # check the time for novelty
            # reset the game env
            # if not env_stop_change_bool and step_idx == int(self.exp_dict_change_position[0]):
            #     env_set_num += 1
            #     self.exp_dict = self.exp_dict_list[env_set_num]
            #
            #     print('self.exp_dict', self.exp_dict)
            #     with HiddenPrints():
            #         self.envs.set_exp(self.exp_dict_list[env_set_num])
            #
            #     exp_dict_change_position.pop(0)
            #     env_stop_change_bool = True if len(exp_dict_change_position) == 0 else False
            #
            #     # reset the game
            #     with HiddenPrints():
            #         reset_array = self.envs.reset()
            #         s, masked_actions, background_actions = [reset_array[i][0] for i in range(len(reset_array))], \
            #                                                 [reset_array[i][1][0] for i in range(len(reset_array))], \
            #                                                 [reset_array[i][1][1] for i in range(len(reset_array))]
            #     loss_train = torch.tensor(0, device=self.device).float()

        self.envs.close()
        # print(self.model.fc_1.weight)
        return self.test_result, True #converge_signal

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        default='/monopoly_simulator_background/config.ini',
                        required=False,
                        help="config_file for env")
    parser.add_argument('--novelty_change_num', type=str,
                        default=None, required=True,
                        help="Novelty price change number")
    parser.add_argument('--novelty_change_begin', type=str,
                        default=None, required=True,
                        help="Novelty price change begin number")
    parser.add_argument('--novelty_introduce_begin', type=str,
                        default=0, required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--seed', type=int,
                        default=0, required=True,
                        help="Env seed")
    parser.add_argument('--upper_path', type=str,
                        default='/media/becky/GNOME-p3', required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--exp_type', type=str,
                        default='kg', required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--tf_stop_num',
                        default=20000, type=int,
                        required=False,
                        help="When to stop using tf")
    parser.add_argument('--device_id',
                        default='0',
                        type=str,
                        required=False,
                        help="GPU id we use")
    parser.add_argument('--adj_path_folder',
                        default='/media/becky/GNOME-p3/KG_rule/matrix_rule/kg_matrix',
                        type=str,
                        required=False,
                        help="The folder has the npz matrix file for adj of game rule/ kg")
    parser.add_argument('--pretrain_model',
                        default=None,
                        type=str,
                        required=False,
                        help="/media/becky/GNOME-p3/monopoly_simulator_background/weights19_1_baseline_seed_9147000.pkl")
    parser.add_argument('--exp_name',
                        default='5_3',
                        type=str,
                        required=False,
                        help="name of adj and log name")
    parser.add_argument('--retrain_type',
                        default='baseline',
                        type=str,
                        required=True,
                        help="gat_part;baseline")
    parser.add_argument('--kg_use',
                        default=True,
                        type=str2bool,
                        help="use kg or not")
    args = parser.parse_args()

    config_data = ConfigParser()
    config_data.read(args.upper_path + args.config_file)

    # set param_list: a list of dictionary
    all_params = {}
    # print('config_data',config_data['hyper'])
    for key in config_data['hyper']:
        v = eval(config_data['hyper'][key])
        if not isinstance(v, tuple):
            all_params[key] = (v,)
        else:
            all_params[key] = v


    def dict_product(d):
        param_list = []
        keys = d.keys()
        for element in product(*d.values()):
            param_list.append(dict(zip(keys, element)))
        return param_list


    param_list = dict_product(all_params)
    exp_dict = dict()
    exp_dict['novelty_num'] = (5, 3)
    exp_dict['novelty_inject_num'] = sys.maxsize
    # exp_dict['exp_type'] = 'kg' if retrain_type != 'baseline' else 'None'
    exp_dict['exp_type'] = args.retrain_type
    for params in param_list:  # run different hyper parameters in sequence
        trainer = MonopolyTrainer_GAT(params,
                                  gameboard=None,
                                  kg_use=args.kg_use,
                                  logger_use=False,
                                  config_file=args.config_file,
                                  test_required=True,
                                  tf_use=True,
                                  pretrain_model=args.pretrain_model,
                                  tf_stop_num=args.tf_stop_num,
                                  exp_name=args.exp_name,
                                  retrain_type=args.retrain_type,
                                  device_id=args.device_id,
                                  seed=0,
                                  adj_path='/media/becky/GNOME-p3/KG_rule/matrix_rule/kg_matrix_no.npy',
                                  exp_dict=exp_dict) #'/media/becky/GNOME-p3/monopoly_simulator_background/weights/no_v3_lr_0.0001_#_107.pkl'
        trainer.train()












