import sys, os
upper_path = os.path.abspath('.').replace('/Evaluation_2/monopoly_simulator_2','')
# upper_path = os.path.abspath('.').replace('/Evaluation_2','')
upper_path_eva = upper_path + '/Evaluation_2/monopoly_simulator_2'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation_2')
sys.path.append(upper_path_eva)
print('upper_path', upper_path, upper_path_eva)
#####################################
from monopoly_simulator_background.gameplay_tf import set_up_board
from monopoly_simulator_2.agent import Agent
from multiprocessing.connection import Client
import A2C_agent_2.RL_agent_v1 as RL_agent_v1
import background_agent_v3
from A2C_agent_2.novelty_detection import KG_OpenIE_eva
from A2C_agent_2.interface_eva import Interface_eva
from A2C_agent_2.novelty_gameboard import detect_card_nevelty, detect_contingent
import torch
from monopoly_simulator_background.vanilla_A2C import *
from monopoly_simulator_2 import action_choices
from configparser import ConfigParser
from A2C_agent_2.KG_A2C import MonopolyTrainer_GAT
import random
import shutil, copy
import logging
from A2C_agent_2.logging_info import log_file_create
import datetime
import A2C_agent_2.agent_helper_functions_agent as agent_helper_functions
from agent import Agent
import socket
import json
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.client_agent_serial')
from scipy import stats
from game_cloning.game_clone import GameClone

# TODO
# √√ 0. path check
#   √√ 0.0 check adj generated correctly before novelty
#   √√ 0.1 check adj generated correctly after novelty
#   √√ 0.3 Make sure every path is correct => adj_path in client after retrain.
#   √√ 0.4 Make sure every path is correct => adj_path in KG-A2C when retraining.

# 1. Retraining setting
#   ✘ 1.1 Time limit, each game is 3 hrs.
#   √ 1.2 Add the winning rate check into KG-A2C functions.
#   √ 1.3 If kg-a2c does not work well in another 10 games (never win), switch to background_agent
#   √ 1.4 Only keep re-retraining when it did not converge.
#   √ 1.5 Retrain from scratch when the gameboard size changed
#   √ 1.6 Retrain every time after novelty detection and no new novelty found, use the one with highest winning rate.
#   ✘ 1.7 Optimize the retraining convergence condition.

# 2. Detect Novelty
#   √ 2.1 detect the bank info
#       √√ 2.1.1 mortgage_percentage
#       √ 2.1.2 property_sell_percentage
#       √ 2.1.3 go_increment
#   √ 2.2 add dice novelty to the game-board after novelty
#   √ 2.3 Card novelty

import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('mk dir =>', path)

def shutdown(serial_dict_to_client, agent):
    '''
    This is an example of how you can return whether novelty was detected or not through the shutdown routine.
    Note: Feel free to modify this code inside this function to whatever works best for you. Using the _agent_memory is just an example of
    how to store novelty detection info when novelty gets detected by your agent. You could store it in whichever way you wish.
    All that we require from you is an int return value that tells us that you detected novelty in this game or not.
    :param serial_dict_to_client:
    :param agent:
    :return: 0->novelty not detected,   1->novelty detected
    '''

    if 'novelty_detection' in agent._agent_memory:
        return agent._agent_memory['novelty_detection']
    else:
        return 0


class ClientAgent(Agent):
    """
    An Agent that can be sent requests from a ServerAgent and send back desired moves. Instantiate the client like you
    would a normal agent, either by passing in the functions to the constructor or by subclassing. The ClientAgent
    can be used for local play as usual, or if the game has been set up with a ServerAgent, call play_remote_game
    to have the client receive game information from the server and send back desired moves.
    """

    def __init__(self, handle_negative_cash_balance, make_pre_roll_move, make_out_of_turn_move, make_post_roll_move,
                 make_buy_property_decision, make_bid, type):
        super().__init__(handle_negative_cash_balance, make_pre_roll_move, make_out_of_turn_move, make_post_roll_move,
                         make_buy_property_decision, make_bid, type)
        self.conn = None
        self.interface = Interface_eva()
        self.kg_change = [[]]
        self.func_history = []
        self.last_func = None
        self.kg_change_bool = False
        self.kg_change_wait = 0
        self.novelty_card = None
        self.novelty_bank = None
        self.gameboard = None
        self.gameboard_ini = None  # gameboard used in the first game of tournament
        self.upper_path = os.path.abspath('.').replace('/Evaluation_2/monopoly_simulator_2', '')
        self.upper_path_eva = self.upper_path + '/Evaluation_2/monopoly_simulator_2'
        self.game_num = 0
        self.seed = random.randint(0, 10000)
        self.retrain_signal = False
        self.change_to_background_wait = 10

        # Read the config
        self.config_file = self.upper_path_eva + '/A2C_agent_2/config.ini'
        self.config_data = ConfigParser()
        self.config_data.read(self.config_file)
        self.hyperparams = self.params_read(self.config_data, 'server')
        self.rule_change_name = self.hyperparams['rule_change_name']
        self.log_file_name = self.hyperparams['log_file_name']

        #kg
        ###set gane board###
        self.kg_use = None
        self.num_players = 4
        self.player_decision_agents = dict()
        # Assign the beckground agents to the players
        name_num = 0
        while name_num < self.num_players:
            name_num += 1
            self.player_decision_agents[
                'player_' + str(name_num)] = Agent(**background_agent_v3.decision_agent_methods)

        # self.gameboard_initial = set_up_board(self.upper_path + '/Evaluation_2/monopoly_game_schema_v1-2.json',
        #                                       self.player_decision_agents,
        #                                       self.num_players)

        self.schema_path = None
        #####################################################
        self.matrix_name = None
        self.entity_name = None
        self.kg = None
        self.no_retrain = False  # set to True when you do not need retrain the agent

        # Save path
        self.folder_path = ''

        # A2C model parameters
        self.state_num = 104
        self.device = torch.device('cpu')
        model_path = self.upper_path_eva + '/A2C_agent_2/0_0_v_gat_part_seed_02400.pkl'
        self.model_path = model_path
        self.model = torch.load(model_path)
        self.logger = logger
        self.adj_path_default = self.upper_path_eva + '/A2C_agent_2/kg_matrix_no.npy'  # TODO, new path after novelty
        self.adj_path = None
        self.adj = None
        self.trainer = None
        self.converge_signal = False  # indicating the convergency of the retraining
        self.best_model = dict()  # a dict recording the best model path
        self.win_rate_after_novelty = None
        self.background_agent_use = False
        self.gat_use = True

        self._agent_memory = dict()

        # game board read
        self.die_count = 2
        self.die_state = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
        self.cards = dict()
        self.minimum_games = 10

        #debug
        self.last_act = None
        self.last_loc = 0
        self.last_func_name = None
        self.call_times = 0

        #novelty
        self.mortgage_percentage = 0.1
        self.mortgage_percentage_list = [0.1]
        self.take_free_mortgage_info = dict()

        self.take_sell_property_info = dict()
        self.property_sell_percentage = 0.5
        self.property_sell_percentage_list = [0.5]

        self.go_increment = 200

        # game cloning
        self.gc = GameClone()
        self.gc_novelty_sig = False
        self.gc_novelty_dict = dict()

    def check_property_sell_percentage(self, cash_after):
        if 'call_time' in self.take_sell_property_info and self.call_times - 1 == self.take_sell_property_info['call_time']:
            if cash_after > self.take_sell_property_info['cash']:
                # print(cash_after , self.take_sell_property_info['cash'] , self.take_sell_property_info['price'])
                rate = (cash_after - self.take_sell_property_info['cash']) / self.take_sell_property_info['price']
                self.property_sell_percentage_list.append(rate)
                # print('self.property_sell_percentage_list', self.property_sell_percentage_list)

                if len(self.property_sell_percentage_list) > 20:
                    self.property_sell_percentage_list.pop(0)
                    property_sell_percentage = stats.mode(self.property_sell_percentage_list)[0][0]
                    if self.property_sell_percentage != property_sell_percentage:
                        if self.kg_use:
                            self.kg_change.append(('property_sell_percentage', self.property_sell_percentage, property_sell_percentage))
                            self.property_sell_percentage = property_sell_percentage

    def check_mortgage_percentage(self, cash_now):
        rate = (self.take_free_mortgage_info['cash'] - cash_now - self.take_free_mortgage_info['mortage']) / self.take_free_mortgage_info['mortage']
        self.mortgage_percentage_list.append(rate)
        # print('self.mortgage_percentage_list', self.mortgage_percentage_list)
        if len(self.mortgage_percentage_list) > 10:
            self.mortgage_percentage_list.pop(0)
            # print('self.mortgage_percentage_list', self.mortgage_percentage_list)
            mortgage_percentage = stats.mode(self.mortgage_percentage_list)[0][0]
            if self.mortgage_percentage != mortgage_percentage:
                if self.kg_use:
                    self.kg_change.append(('mortgage_percentage', self.mortgage_percentage, mortgage_percentage))
                    self.mortgage_percentage = mortgage_percentage


    def kg_run(self, current_gameboard, card_board, game_num):
        """
        Run the knowledge graph and novelty detection here
        :param current_gameboard:
        :return:
        """
        self.logger.debug('Run the knowledge graph to learn the rule')
        self.interface.set_board(current_gameboard)
        self.interface.get_logging_info_once(current_gameboard, self.folder_path+self.log_file_name)
        self.interface.get_logging_info(current_gameboard, self.folder_path+self.log_file_name)

        #######Add kg here######
        self.kg.set_gameboard(current_gameboard)
        self.kg.build_kg_file(self.folder_path+self.log_file_name, level='rel', use_hash=False)
        self.kg.card_board.read_card_board(card_board, game_num)
        self.cards = copy.deepcopy(self.kg.card_board.cards)



        # self.kg.save_file(self.kg.kg_rel, self.kg_rel_path)
        # print('kg_run!!!!!!!!')
        # if 'is rented-0-house at' in self.kg.kg_rel.keys():
        #     if 'Indiana-Avenue' in self.kg.kg_rel['is rented-0-house at'].keys():
        #         print(self.kg.kg_rel['is rented-0-house at']['Indiana-Avenue'])

        # print(self.kg.kg_change)

        ####no need for evaluation####
        # self.kg.dict_to_matrix()
        # self.kg.save_matrix()
        # self.kg.save_vector()
        ###############################

        # if self.kg_change_wait > 0:
        #     self.kg_change_wait += 1
        #     self.logger.debug('Novelty Firstly Detected ' + str(self.kg_change_wait) + ' times ago.')
        #     self.logger.debug('New Novelty Detected as ' + str(self.kg_change))

        self.kg.save_matrix()
        # Visulalize the novelty change in the network
        self.kg_change_bool = False

        if self.kg_use and self.kg.kg_change != self.kg_change[-1]:
            self.kg_change_wait = 1
            self.kg_change.append(self.kg.kg_change[:])
            self.logger.debug('Novelty Detected as ' + str(self.kg_change))
            print('Novelty Detected as ', str(self.kg_change))
            self.retrain_signal = True  # if self.kg_change_bool == False else False
            self.converge_signal = False  # mean we need to retrain.
            self.kg_change_bool = True
            self.best_model = dict()
            if self.win_rate_after_novelty == None:
                self.win_rate_after_novelty = []

            # log
            if self.retrain_signal == True:
                self.logger.debug('Retrain signal is True now, it will retrain the NN before next game!')
            else:
                self.logger.debug('Retrain happened before, no need to retrain again!')

            # ini_current_gameboard = self.initialize_gameboard(current_gameboard)
            # model_retrained_path = self.retrain_from_scratch(ini_current_gameboard)
            # self.model = torch.load(model_retrained_path)
            # self.state_num = len(self.interface.board_to_state(current_gameboard))  # reset the state_num size
            # self.kg_change_bool = True
            # self.kg_change_wait = 0

        # game cloning detects novelty
        if self.gc_novelty_sig:
            self.retrain_signal = True
            self.logger.debug('Novelty Detected by Game Cloning')
            self.logger.debug('Novelty Detected as ' + str(self.gc_novelty_dict))
            self.logger.debug('Retrain signal is True now, it will retrain the NN before next game!')

    def novelty_board_detect(self, current_gameboard):
        self.die_count = (current_gameboard['die_sequence'][0])

    def load_adj(self):
        """
        load the relationship matrix
        :param path:
        :return:
        """
        if os.path.exists(self.adj_path):
            print('adj_use', self.adj_path)
            self.adj = np.load(self.adj_path)
            adj_return = np.zeros((self.adj.shape[0], self.adj.shape[0]))
            for i in range(self.adj.shape[1] // self.adj.shape[0]):
                adj_return += self.adj[:,self.adj.shape[0] * i:self.adj.shape[0] * (i+1)]
            self.adj = adj_return

        else:
            print('adj_use', self.adj_path_default)
            self.adj = np.load(self.adj_path_default)
            adj_return = np.zeros((self.adj.shape[0], self.adj.shape[0]))
            for i in range(self.adj.shape[1] // self.adj.shape[0]):
                adj_return += self.adj[:, self.adj.shape[0] * i:self.adj.shape[0] * (i + 1)]
            self.adj = adj_return

    def params_read(self, config_data, key_word):
        params = {}
        for key in config_data[key_word]:
            v = eval(config_data[key_word][key])
            params[key] = v
        return params

    def define_path(self, path=''):
        # Define the folder place the saved weights
        self.folder_path = os.path.join(self.upper_path_eva, 'A2C_agent_2/weights', path)

        self.matrix_name = os.path.join(self.folder_path, 'matrix.npy')
        self.entity_name = os.path.join(self.folder_path, 'entity.json')
        self.adj_path = self.matrix_name

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        else:
            shutil.rmtree(self.folder_path)
            os.makedirs(self.folder_path)

        # Create logging info for this file
        self.logger = log_file_create(os.path.join(self.folder_path, 'log_client.log'))
        self.logger.debug('Define the folder path as ' + self.folder_path)

        # print('self.folder_path', self.folder_path)

    def make_post_roll_move_agent(self, serial_dict_to_client):
        player_name = serial_dict_to_client['player']
        current_gameboard = serial_dict_to_client['current_gameboard']
        player = current_gameboard['players'][player_name]
        allowable_moves = serial_dict_to_client['allowable_moves']
        code = serial_dict_to_client['code']
        return_to_server_dict = dict()

        # check go increment
        for i in current_gameboard['history']:
            if i['function'] == 'update_player_position' and 'amount' in i['param'] and i['param']['description'] == 'go increment':
                go_increment = int(i['param']['amount'])
                if self.go_increment != go_increment:
                    self.kg_change.append(('go increment', self.go_increment, go_increment))
                print('self.go_increment', self.go_increment)

        # call A2C agent##########################################
        self.gameboard = copy.deepcopy(current_gameboard)
        self.interface.get_logging_info_once(current_gameboard, self.folder_path+self.log_file_name)
        s = self.interface.board_to_state(current_gameboard)
        s = s.reshape(1, -1)
        s = torch.tensor(s, device=self.device).float()
        if self.gat_use:
            self.load_adj()  # call kg-matrix
            s = self.model.forward(s, self.adj)
        prob = self.model.actor(s)
        action = Categorical(prob).sample().cpu().numpy()
        action = action[0]
        actions_vector = self.interface.action_num2vec(action)
        move_actions = self.interface.vector_to_actions(current_gameboard, player, actions_vector)
        ##########################################################

        move_action = [i[0] for i in move_actions]
        space_action = [i[1] for i in move_actions]

        current_location = current_gameboard['location_sequence'][player['current_position']]

        if 'buy_property' in move_action:
            if "buy_property" in allowable_moves:
                params = dict()
                params['player'] = 'player_1'
                params['asset'] = current_location
                params['current_gameboard'] = "current_gameboard"
                if code == -1:
                    return_to_server_dict['function'] = "concluded_actions"
                    return_to_server_dict['param_dict'] = dict()
                    return return_to_server_dict
                self._agent_memory['previous_action'] = "buy_property"
                return_to_server_dict['function'] = "buy_property"
                return_to_server_dict['param_dict'] = params
                self.last_act = "buy_property"
                # debug - if the buy works####
                # print('s_ini', s_ini[0][s_ini[0][40:80].tolist().index(1)])
                # if 1 in s_ini[0][40:80].tolist():
                #     self.last_loc = s_ini[0][40:80].tolist().index(1)
                # ##############################

                return return_to_server_dict

        current_location = current_gameboard['locations'][current_location]
        # mortgage
        if agent_helper_functions.will_property_complete_set(player, current_location, current_gameboard):
            to_mortgage = agent_helper_functions.identify_potential_mortgage(player, current_location['price'],current_gameboard, True)
            if to_mortgage:
                params = dict()
                params['player'] = "player_1"
                params['asset'] = to_mortgage['name']
                params['current_gameboard'] = "current_gameboard"
                logger.debug('player_1' + ': I am attempting to mortgage property ' + params['asset'])
                self._agent_memory['previous_action'] = "mortgage_property"
                return_to_server_dict['function'] = "mortgage_property"
                return_to_server_dict['param_dict'] = params
                self.last_act = "mortgage_property"
                return return_to_server_dict

        return_to_server_dict['function'] = "concluded_actions"
        return_to_server_dict['param_dict'] = dict()
        self._agent_memory['previous_action'] = "concluded_actions"
        self.last_act = "concluded_actions"

        return return_to_server_dict

    def initialize_gameboard(self, gameboard):
        """
        Intialize the used gameboard
        :param gameboard:
        :return:
        """
        self.schema_path = self.folder_path + '/schema.json'

        game_schema = json.load(open(self.upper_path + '/Evaluation_2/monopoly_game_schema_v1-2.json', 'r'))

        # location
        game_schema['locations']['location_count'] = len(gameboard['location_sequence'])
        game_schema['location_sequence'] = gameboard['location_sequence']
        location_states = []
        for name in game_schema['location_sequence']:
            for key in ['is_mortgaged', 'house_rent_dict', 'railroad_dues', 'die_multiples']:
                if key in gameboard['locations'][name].keys():
                    del gameboard['locations'][name][key]
            if 'perform_actiom' in gameboard['locations'][name].keys():
                gameboard['locations'][name]['perform_action'] = copy.deepcopy(gameboard['locations'][name]['perform_actiom'])
                del gameboard['locations'][name]['perform_actiom']

            location_states.append(gameboard['locations'][name])
            if name == 'Go':
                game_schema['go_position'] = int(gameboard['locations'][name]['start_position'])
        game_schema['locations']['location_states'] = location_states

        # go_increment

        game_schema["go_increment"] = self.go_increment

        # die add novelty from kg
        if self.kg_change and 'Dice' in self.kg_change[-1]:
            for change in self.kg_change[-1][-1]:
                if 'State' in change:
                    game_schema['die']['die_count'] = len(change[1])
                    game_schema['die']['die_state'] = change[1]

        #cards
        community_chest_list = []
        for name in self.cards['community_chest']:
            self.cards['community_chest'][name]['num'] = 1
            community_chest_list.append(self.cards['community_chest'][name])
        game_schema['cards']['community_chest']['card_states'] = community_chest_list
        chance_list = []
        for name in self.cards['chance']:
            self.cards['chance'][name]['num'] = 1
            chance_list.append(self.cards['chance'][name])
        game_schema['cards']['chance']['card_states'] = chance_list

        with open(self.schema_path, 'w') as f:
            json.dump(game_schema, f)

        ini_gameboard = set_up_board(self.schema_path,self.player_decision_agents, self.num_players)

        # bank info
        ini_gameboard['bank'].mortgage_percentage = self.mortgage_percentage
        ini_gameboard['bank'].property_sell_percentage = self.property_sell_percentage

        #dice type
        if self.kg_change and 'Dice' in self.kg_change[-1]:
            for change in self.kg_change[-1][-1]:
                if 'Type' in change:
                    type = change[1]
                    for idx, die in enumerate(ini_gameboard['dies']):
                        die.die_state_distribution = type[idx].lower()

        return ini_gameboard

    def retrain_ini(self, gameboard, type_retrain='size'):
        """
        set the trainer here
        :param gameboard:
        :return:
        """
        if self.no_retrain:
            return
        # set exp dict
        exp_dict = dict()
        exp_dict['novelty_num'] = (0, 0)
        exp_dict['novelty_inject_num'] = sys.maxsize
        exp_dict['exp_type'] = '0_0'

        # Begin retraining
        model_path = self.model_path
        config_data = ConfigParser()
        config_data.read(self.config_file)
        params = self.params_read(config_data, 'hyper')
        len_vocab = 52 + len(gameboard['location_sequence'])

        if type_retrain == 'size':
            trainer = MonopolyTrainer_GAT(params,
                                          gameboard=gameboard,
                                          kg_use=False,
                                          logger_use=False,
                                          config_file='config.ini',
                                          test_required=True,  # change to False if you do not need any test check anymore.
                                          tf_use=False,
                                          pretrain_model=None,
                                          retrain_type='baseline',
                                          device_id='-1',
                                          seed=0,
                                          adj_path=None,
                                          exp_dict=exp_dict,
                                          len_vocab=len_vocab)
            self.gat_use = False
        else:
            trainer = MonopolyTrainer_GAT(params,
                                          gameboard=gameboard,
                                          kg_use=True,
                                          logger_use=False,
                                          config_file='config.ini',
                                          test_required=True,
                                          # change to False if you do not need any test check anymore.
                                          tf_use=False,
                                          pretrain_model=None,
                                          retrain_type='gat_part',
                                          device_id='-1',
                                          seed=0,
                                          adj_path=self.adj_path,
                                          exp_dict=exp_dict,
                                          len_vocab=len_vocab)

        self.best_model = dict()

        if os.path.exists(upper_path_eva + "/A2C_agent_2/logs/kg_matrix_0_0.npy"):
            os.remove(upper_path_eva + "/A2C_agent_2/logs/kg_matrix_0_0.npy")

        return trainer

    def retrain_from_scratch(self, gameboard, game_num):
        """
        Retrain the model and update the model used
        :param gameboard:
        :return: adj_path:
        """
        if self.no_retrain:
            self.logger.info('No retrain is needed')
            return self.adj_path

        print('REtraining ......')
        start_time = datetime.datetime.now()
        self.logger.info('Retrain start!!! It will save to ' + self.folder_path)

        # offline training setting ##############################
        params = self.params_read(self.config_data, 'hyper')
        params['save_path'] = self.folder_path + '/'

        save_path = self.folder_path + '/' + str(game_num) + '/'
        mkdir(self.folder_path + '/' + str(game_num))

        self.trainer.set_gameboard(gameboard=gameboard,
                                   save_path=save_path)

        # #######################################################

        # retrain the model #####################################
        test_result, self.converge_signal = self.trainer.train()
        # #######################################################

        # calculate the best performance one ####################
        self.logger.info('Retrain results => ' + str(test_result))

        win_rate = test_result['winning_rate']
        win_rate.reverse()
        opt = len(win_rate) - win_rate.index(max(win_rate))
        model_retrained_path = save_path + str(opt) + '.pkl'

        if 'win_rate' in self.best_model:
            if self.best_model['win_rate'] > max(win_rate):
                model_retrained_path = self.best_model['model_path']
            else:
                self.best_model['win_rate'] = max(win_rate)
                self.best_model['model_path'] = model_retrained_path
        else:
            self.best_model['win_rate'] = max(win_rate)
            self.best_model['model_path'] = model_retrained_path

        # #######################################################

        # calculate the time used ###############################
        end_time = datetime.datetime.now()
        time_retrain = (end_time - start_time).seconds / 60
        self.logger.info('Retrain use ' + str(time_retrain)  + 'mins')
        self.logger.info('The ' + str(opt) + 'steps is the best, we will use ' + model_retrained_path)
        # #######################################################

        # update model ##########################################
        self.model = torch.load(model_retrained_path)
        self.kg_change_wait = 0
        # #######################################################
        return self.trainer.adj_path

    def change_to_background(self, player_board):
        if self.win_rate_after_novelty != None:
            if player_board['player_1']['status'] == 'lost':
                self.win_rate_after_novelty.append(0)
            else:
                self.win_rate_after_novelty.append(1)

            if len(self.win_rate_after_novelty) >= self.change_to_background_wait:
                if sum(self.win_rate_after_novelty[-1 * self.change_to_background_wait:]) == 0:
                    self.background_agent_use = True

    def play_remote_game(self, address=('localhost', 6010), authkey=b"password"):
        """
        Connects to a ServerAgent and begins the loop of waiting for requests and responding to them.
        @param address: Tuple, the address and port number. Defaults to localhost:6000
        @param authkey: Byte string, the password used to authenticate the client. Must be same as server's authkey.
            Defaults to "password"
        """
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((address[0], address[1]))
        # self.conn = Client(address, authkey=authkey)

        result = None
        ini_sig = True  # if it is the first time to setup trainer.
        board_size_changed_sig = False
        while True:
            # receive the info form server###########################
            data_from_server = self.conn.recv(200000)
            data_from_server = data_from_server.decode("utf-8")

            data_dict_from_server = json.loads(data_from_server)
            func_name = data_dict_from_server['function']
            self.call_times += 1

            # game cloning novelty check up
            if func_name in ['make_buy_property_decision', 'make_bid', 'make_pre_roll_move', 'make_out_of_turn_move', 'make_post_roll_move'] and \
                not self.gc_novelty_sig:
                self.gc_novelty_sig, novelty_dict = self.gc.gc_detect_novelty(data_from_server)
                if self.gc_novelty_sig:
                    self.gc_novelty_dict.update(novelty_dict)

            if self.last_func_name == 'handle_negative_cash_balance' and 'current_gameboard' in data_dict_from_server:
                self.check_property_sell_percentage(data_dict_from_server['current_gameboard']['players']['player_1']['current_cash'])
                # print('after', data_dict_from_server['current_gameboard']['players']['player_1']['current_cash'])
            # #######################################################

            # When the tournament begins, we need to define the folder
            if func_name == "start_tournament":
                self.define_path() #data_dict_from_server['path'])  # save all the file in this folder
                self.kg_use = True #if data_dict_from_server['info'] == 'w/' else False
                self.logger.info('Tournament starts!')
                result = 1
            # #######################################################

            # Before simulating each game, we have to make sure if we need retrain the network
            elif func_name == "startup":
                self.call_times = 1
                # 1. Clear interface history and set the init for interface#########
                self.interface.clear_history(os.path.join(self.folder_path,self.log_file_name[1:]))
                self.interface.set_board(data_dict_from_server['current_gameboard'])
                print('sequence =>', data_dict_from_server['current_gameboard']['location_sequence'])
                s = self.interface.board_to_state(data_dict_from_server['current_gameboard'])
                self.game_num += 1  # update number of games simulated
                self.logger.info(str(self.game_num) + ' th game starts!')
                # ##################################################################

                # 2. We need to set up kg before the first game#####################
                if self.game_num == 1:
                    self.gameboard_ini = data_dict_from_server['current_gameboard']
                    self.kg = KG_OpenIE_eva(self.gameboard_ini,
                                                self.matrix_name,
                                                self.entity_name,
                                                config_file=self.config_file)
                # ###################################################################

                # 3. If board size changed, before the game, we need to call retrain#
                print('self.state_num', self.state_num, len(s))
                if self.state_num != len(s) or self.retrain_signal:
                    if self.state_num != len(s):
                        self.logger.info('detect the novelty as the board size change')

                    if self.win_rate_after_novelty == None:  # begin to record the game winning rate
                        self.win_rate_after_novelty = []

                    if self.state_num != len(s):
                        board_size_changed_sig = True
                    retrain_type = 'size' if board_size_changed_sig else 'novelty'
                    ini_current_gameboard = self.initialize_gameboard(data_dict_from_server['current_gameboard'])  #TODO

                    # 3.1 set up trainer:
                    # if ini_sig or board_size_changed_sig or self.kg_change_bool: # or (self.converge_signal and not self.kg_change_bool):  # 1st time or board size changed, we need to initialize trainer.
                    self.trainer = self.retrain_ini(ini_current_gameboard, retrain_type)
                        # ini_sig = False

                    # 3.2 begin retraining
                    adj_path = self.retrain_from_scratch(ini_current_gameboard, self.game_num)

                    self.state_num = len(s)  # reset the state_num size

                    # self.kg_change_bool = True
                    self.retrain_signal = False if self.converge_signal else True  # when converge, stop retraining
                    # self.kg_change_wait = 0
                    # self.adj_path = adj_path

            # #############################################################################

            # When each game ends, we run the KG, but we don not shutdown the connection
            elif func_name == 'shutdown':
                serial_dict_to_client = data_dict_from_server
                if 'players' in serial_dict_to_client:
                    self.change_to_background(serial_dict_to_client['players'])
                if 'cards' in serial_dict_to_client:
                    self.kg_run(self.gameboard, serial_dict_to_client['cards'], self.game_num)
                result = shutdown(serial_dict_to_client, self)
                self.logger.info(str(self.game_num) + ' th game stops!')

            # When calling agent to make decision
            elif func_name == 'make_post_roll_move':
                serial_dict_to_client = data_dict_from_server
                if self.background_agent_use:
                    server_result, self._agent_memory = self.make_post_roll_move(serial_dict_to_client, self._agent_memory, self.go_increment)
                    serial_dict_to_server = dict()
                    serial_dict_to_server['function'] = server_result[0]
                    serial_dict_to_server['param_dict'] = server_result[1]
                    result = serial_dict_to_server
                else:
                    print('serial_dict_to_server', data_dict_from_server['current_gameboard']['cards']['picked_chance_card_details'])
                    result = self.make_post_roll_move_agent(serial_dict_to_client)
                    # server_result, self._agent_memory = self.make_post_roll_move(serial_dict_to_client,
                    #                                                              self._agent_memory)
                    # serial_dict_to_server = dict()
                    # serial_dict_to_server['function'] = server_result[0]
                    # serial_dict_to_server['param_dict'] = server_result[1]
                    # result = serial_dict_to_server

            elif func_name == 'make_out_of_turn_move':
                serial_dict_to_client = data_dict_from_server
                self.gameboard = serial_dict_to_client['current_gameboard']
                if self.gameboard['history'][-1]['function'] == 'free_mortgage':
                    self.check_mortgage_percentage(self.gameboard['players']['player_1']['current_cash'])
                server_result, self._agent_memory = self.make_out_of_turn_move(serial_dict_to_client, self._agent_memory, self.mortgage_percentage, self.go_increment)
                serial_dict_to_server = dict()
                serial_dict_to_server['function'] = server_result[0]
                serial_dict_to_server['param_dict'] = server_result[1]
                if serial_dict_to_server['function'] =='free_mortgage':
                    self.take_free_mortgage_info['mortage'] = self.gameboard['locations'][serial_dict_to_server['param_dict']['asset']]['mortgage']
                    self.take_free_mortgage_info['cash'] = self.gameboard['players']['player_1']['current_cash']
                result = serial_dict_to_server

            elif func_name == 'make_pre_roll_move':
                serial_dict_to_client = data_dict_from_server
                result,self._agent_memory = self.make_pre_roll_move(serial_dict_to_client, self._agent_memory)

            elif func_name == 'make_bid':
                serial_dict_to_client = data_dict_from_server
                result = self.make_bid(serial_dict_to_client)

            elif func_name == 'make_buy_property_decision':
                serial_dict_to_client = data_dict_from_server
                result = self.make_buy_property_decision(serial_dict_to_client, self.go_increment)

            elif func_name == 'handle_negative_cash_balance':
                serial_dict_to_client = data_dict_from_server
                server_result, self._agent_memory = self.handle_negative_cash_balance(serial_dict_to_client, self._agent_memory, self.property_sell_percentage, self.mortgage_percentage)
                serial_dict_to_server = dict()
                serial_dict_to_server['function'] = server_result[0]
                serial_dict_to_server['param_dict'] = server_result[1]
                if server_result[0] == "sell_property":
                    self.take_sell_property_info['cash'] = data_dict_from_server['current_gameboard']['players']['player_1']['current_cash']
                    self.take_sell_property_info['price'] = data_dict_from_server['current_gameboard']['locations'][server_result[1]['asset']]['price']
                    self.take_sell_property_info['call_time'] = self.call_times
                    # print("!!!!!!sell_property")
                    # # print(data_dict_from_server['current_gameboard']['players']['player_1']['current_cash'])
                    # print('cash', data_dict_from_server['current_gameboard']['players']['player_1']['current_cash'])
                    # print('price', data_dict_from_server['current_gameboard']['locations'][server_result[1]['asset']]['price'])

                result = serial_dict_to_server

            # Send we will close the connection now back to server
            elif func_name == "end_tournament":
                result = 1
                self.logger.info('Tournament Finished!')

            else:
                serial_dict_to_client = data_dict_from_server
                result = getattr(self, func_name)(serial_dict_to_client)

            self.last_func_name = func_name
            # Send the results back to server agent
            if isinstance(result, int):
                self.conn.sendall(bytes(str(result), encoding="utf-8"))
            elif isinstance(result, float):
                self.conn.sendall(bytes(str(result), encoding="utf-8"))
            elif isinstance(result, bool):
                self.conn.sendall(bytes(str(result), encoding="utf-8"))
            else:  #dictionary
                json_serial_return_to_server = json.dumps(result)
                self.conn.sendall(bytes(json_serial_return_to_server, encoding="utf-8"))


            # Close connection after each tournament
            if func_name == "end_tournament":
                self.conn.close()
                break


def main():
    client = ClientAgent(**RL_agent_v1.decision_agent_methods)
    client.play_remote_game()


if __name__ == "__main__":
    main()

