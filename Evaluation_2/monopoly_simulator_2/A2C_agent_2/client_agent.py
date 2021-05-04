import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation_2/monopoly_simulator','')
# upper_path = os.path.abspath('.').replace('/Evaluation','')
upper_path_eva = upper_path + '/Evaluation_2/monopoly_simulator'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation_2')
sys.path.append(upper_path_eva)
#####################################

from monopoly_simulator.agent import Agent
from multiprocessing.connection import Client
import A2C_agent.RL_agent_v1 as RL_agent_v1
from A2C_agent.novelty_detection import KG_OpenIE_eva
from A2C_agent.interface_eva import Interface_eva
from A2C_agent.novelty_gameboard import detect_card_nevelty, detect_contingent
import torch
from monopoly_simulator_background.vanilla_A2C import *
from monopoly_simulator import action_choices
from configparser import ConfigParser
from A2C_agent.KG_A2C import MonopolyTrainer_GAT
import random
import shutil, copy
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.A2C_agent.client_agent')
from A2C_agent.logging_info import log_file_create
import datetime

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
        self.kg = KG_OpenIE_eva()
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
        self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator', '')
        self.upper_path_eva = self.upper_path + '/Evaluation/monopoly_simulator'
        self.game_num = 0
        self.seed = random.randint(0, 10000)
        self.retrain_signal = False

        #Read the config
        self.config_file = self.upper_path_eva + '/A2C_agent/config.ini'
        self.config_data = ConfigParser()
        self.config_data.read(self.config_file)
        self.hyperparams = self.params_read(self.config_data, 'server')
        self.rule_change_name = self.hyperparams['rule_change_name']
        self.log_file_name = self.hyperparams['log_file_name']

        # Save path
        self.folder_path = None

        #A2C model parameters
        self.state_num = 90
        self.device = torch.device('cpu')
        model_path = self.upper_path_eva + '/A2C_agent/weights/Original_cpu_2.pkl'
        self.model = torch.load(model_path)
        self.logger = None
        self.adj_path = self.upper_path_eva + '/A2C_agent/kg_matrix_no.npy'


    def define_path(self, path):
        # Define the folder place the saved weights
        self.folder_path = self.upper_path_eva + '/A2C_agent/weights/' + path

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        else:
            shutil.rmtree(self.folder_path)
            os.makedirs(self.folder_path)

        # Create logging info for this file
        self.logger = log_file_create(self.folder_path + '/log_client.log')
        self.logger.debug('Define the folder path as ' + self.folder_path)

    def initialize_gameboard(self, gameboard):
        """
        Intialize the used gameboard
        :param gameboard:
        :return:
        """
        gameboard_keys = ['bank', 'jail_position', 'railroad_positions', 'utility_positions', 'go_position', 'go_increment', 'location_objects', 'location_sequence', 'color_assets', 'dies', 'chance_cards', 'community_chest_cards', 'chance_card_objects', 'community_chest_card_objects','players', 'type']
        ini_gameboard = dict()
        for key in gameboard_keys:
            ini_gameboard[key] = gameboard[key]
        ini_gameboard['current_die_total'] = 0
        ini_gameboard['die_sequence'] = []
        ini_gameboard['history'] = dict()
        ini_gameboard['history'] = dict()
        ini_gameboard['history']['function'] = list()
        ini_gameboard['history']['param'] = list()
        ini_gameboard['history']['return'] = list()
        ini_gameboard['picked_community_chest_cards'] = []
        ini_gameboard['picked_chance_cards'] = []
        for i, player in enumerate(['players']):
            ini_gameboard['players'][i].agent = None
        return ini_gameboard


    def kg_run(self, current_gameboard):
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

        if self.kg_change_wait > 0:
            self.kg_change_wait += 1
            self.logger.debug('Novelty Firstly Detected ' + str(self.kg_change_wait) + ' times ago.')
            self.logger.debug('New Novelty Detected as ' + str(self.kg_change))

        # Visulalize the novelty change in the network
        if self.kg.kg_change != self.kg_change[-1]:
            self.kg_change_wait = 1
            self.kg_change.append(self.kg.kg_change[:])
            self.logger.debug('Novelty Detected as ' + str(self.kg_change))

        # Only when never retrained and detect novelty, will call retrain
        if self.kg_change_wait > 3 and self.kg_change_bool == False:
            self.retrain_signal = True if self.kg_change_bool == False else False

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


    def params_read(self, config_data, key_word):
        params = {}
        for key in config_data[key_word]:
            v = eval(config_data[key_word][key])
            params[key] = v
        return params


    def retrain_from_scratch(self, gameboard):
        """
        Retrain the model and save the model
        :param gameboard:
        :return:
        """
        start_time = datetime.datetime.now()

        self.logger.info('Retrain start!!! It will save to ' + self.folder_path)

        # Retrain the network
        params = self.params_read(self.config_data, 'hyper')
        params['save_path'] = self.folder_path
        # set exp dict
        exp_dict = dict()
        exp_dict['novelty_num'] = (0, 0)
        exp_dict['novelty_inject_num'] = sys.maxsize
        exp_dict['exp_type'] = '0_0'

        trainer = MonopolyTrainer_GAT(params,
                                      gameboard=gameboard,
                                      kg_use=True,
                                      logger_use=False,
                                      config_file='config.ini',
                                      test_required=False,
                                      tf_use=False,
                                      retrain_type='gat_part',
                                      device_id='-1',
                                      adj_path=self.adj_path,
                                      exp_dict=exp_dict)

        trainer.train()
        self.logger.info('Retrain results => ' + str(trainer.test_result))

        win_rate = trainer.test_result['winning_rate']
        win_rate.reverse()
        opt = len(win_rate) - win_rate.index(max(win_rate))
        model_retrained_path = self.folder_path + '/cpu_v3_lr_' + str(params['learning_rate']) + '_#_' + str(opt) + '.pkl'

        end_time = datetime.datetime.now()
        time_retrain = (end_time - start_time).seconds / 60
        self.logger.info('Retrain use ' + str(time_retrain)  + 'mins')
        self.logger.info('The ' + str(opt) + 'steps is the best, we will use ' + model_retrained_path)
        self.kg_change_wait = 0

        return model_retrained_path, trainer.adj_path




    def play_remote_game(self, address=('localhost', 6001), authkey=b"password"):
        """
        Connects to a ServerAgent and begins the loop of waiting for requests and responding to them.
        @param address: Tuple, the address and port number. Defaults to localhost:6000
        @param authkey: Byte string, the password used to authenticate the client. Must be same as server's authkey.
            Defaults to "password"
        """

        self.conn = Client(address, authkey=authkey)
        while True:
            func_name, args = self.conn.recv()  # Receive the signal

            # # When the game starts, we clear all the history
            # if func_name == 'start_tournament':
            #     self.clear_weights()
            #     result = 1

            # When the tournament begins, we need to
            if func_name == "start_tournament":
                self.define_path(args)
                self.logger.info('Tournament starts!')
                result = 1

            # Before simulating each game, we have to make sure if we need retrain the network
            elif func_name == "startup": # args = (current_gameboard, indicator)
                # Clear interface history and set the init for interface
                self.interface.clear_history(self.folder_path+self.log_file_name)
                self.interface.set_board(args[0])  # args[0] is current_gameboard
                s = self.interface.board_to_state(args[0])
                print('s', s)
                self.game_num += 1  # number of games simulated
                self.logger.info(str(self.game_num) + ' th game starts!')

                # We need to have a baseline of the gameboard
                if self.game_num == 1:
                    self.gameboard_ini = args[0]

                # Detect card type and number change with comparing gameboard
                if self.novelty_card == None:
                    self.novelty_card = detect_card_nevelty(args[0], self.gameboard_ini)
                if self.novelty_bank == None:
                    self.novelty_bank = detect_contingent(args[0], self.gameboard_ini)

                if self.novelty_card:
                    self.kg_change.append(self.novelty_card)
                    self.novelty_card = []
                if self.novelty_bank:
                    self.kg_change.append(self.novelty_bank)
                    self.novelty_bank = []

                # if board size changed, before the game, we need to retrain the NN
                if self.state_num != len(s) or self.retrain_signal:
                    ini_current_gameboard = self.initialize_gameboard(args[0])
                    model_retrained_path, adj_path = self.retrain_from_scratch(ini_current_gameboard)

                    self.model = torch.load(model_retrained_path)
                    self.state_num = len(self.interface.board_to_state(args[0]))  # reset the state_num size
                    self.kg_change_bool = True
                    self.retrain_signal = False
                    self.kg_change_wait = 0
                    self.adj_path = adj_path

                result = getattr(self, func_name)(*args)

            # When each game ends, we run the KG, but we don not shutdown the connection
            elif func_name == 'shutdown':
                result = 1
                self.kg_run(self.gameboard)
                self.logger.info(str(self.game_num) + ' th game stops!')

            # When calling agent to make decision
            elif func_name == 'make_post_roll_move':  # args = (player, current_gameboard, allowable_moves, code)
                result = self.make_post_roll_move_agent(args)

            # Send we will close the connection now back to server
            elif func_name == "end_tournament":
                result = 1
                self.logger.info('Tournament Finished!')

            else:
                result = getattr(self, func_name)(*args)

            # Send the results back to server agent
            self.conn.send(result)

            # Close connection after each tournament
            if func_name == "end_tournament":

                # Write to history
                file = open(self.folder_path + self.rule_change_name, "a")
                file.write(str(self.seed) + str(self.kg_change) + ' \n')
                file.close()
                self.logger.info('Novelty of game saved to ' + self.folder_path + self.rule_change_name)

                self.conn.close()
                break




    def make_post_roll_move_agent(self, args):
        player, current_gameboard, allowable_moves, code = args
        self.gameboard = copy.deepcopy(current_gameboard)

        # self.interface.set_board(current_gameboard)
        # a.save_history(current_gameboard, save_path='/media/becky/Evaluation/GNOME-p3/monopoly_simulator/loc_history.pickle')

        # if player.assets == None:
        # a.save_bool(0)
        self.interface.get_logging_info_once(current_gameboard, self.folder_path+self.log_file_name)

        s = self.interface.board_to_state(current_gameboard)

        # if self.state_num != len(s):
        #     ini_current_gameboard = self.initialize_gameboard(args[1])
        #     model_retrained_path = self.retrain_from_scratch(ini_current_gameboard)
        #     self.model = torch.load(model_retrained_path)
        #     self.state_num = len(self.interface.board_to_state(args[1]))  # reset the state_num size
        #     self.kg_change_bool = True
        #     self.kg_change_wait = 0


        s = s.reshape(1, -1)
        # print(s.shape)
        s = torch.tensor(s, device=self.device).float()
        prob = self.model.actor(s)
        action = Categorical(prob).sample().cpu().numpy()
        action = action[0]
        # print(action)
        actions_vector = self.interface.action_num2vec(action)
        move_actions = self.interface.vector_to_actions(current_gameboard, player, actions_vector)
        #####################

        move_action = [i[0] for i in move_actions]
        space_action = [i[1] for i in move_actions]

        current_location = current_gameboard['location_sequence'][player.current_position]

        if action_choices.buy_property in move_action:
            if action_choices.buy_property in allowable_moves:
                params = dict()
                params['player'] = player
                params['asset'] = current_location
                params['current_gameboard'] = current_gameboard
                if code == -1:
                  return (action_choices.concluded_actions, dict())
                player.agent._agent_memory['previous_action'] = action_choices.buy_property
                return (action_choices.buy_property, params)

        # improve property
        if action_choices.improve_property in move_action:  # beef up full color sets to maximize rent potential.
            if action_choices.improve_property in allowable_moves:
                params = dict()
                params['player'] = player
                params['asset'] = space_action[0]
                params['current_gameboard'] = current_gameboard
                player.agent._agent_memory['previous_action'] = action_choices.improve_property
                return (action_choices.improve_property, params)

        # free-mortgage
        if action_choices.free_mortgage in move_action:
            if action_choices.free_mortgage in allowable_moves:
                # free mortgages till we can afford it. the second condition should not be necessary but just in case.
                params = dict()
                params['player'] = player
                params['asset'] = space_action[0]
                params['current_gameboard'] = current_gameboard
                player.agent._agent_memory['previous_action'] = action_choices.free_mortgage
                return (action_choices.free_mortgage, params)

        # mortgage
        if action_choices.mortgage_property in move_action:
            if action_choices.mortgage_property in allowable_moves:
                params = dict()
                params['player'] = player
                params['asset'] = space_action[0]
                params['current_gameboard'] = current_gameboard
                player.agent._agent_memory['previous_action'] = action_choices.mortgage_property
                return (action_choices.mortgage_property, params)
        return (action_choices.concluded_actions, dict())



def main():
    client = ClientAgent(**RL_agent_v1.decision_agent_methods)
    client.play_remote_game()


if __name__ == "__main__":
    main()