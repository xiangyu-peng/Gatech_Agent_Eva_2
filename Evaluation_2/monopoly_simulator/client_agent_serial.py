import sys, os
upper_path = os.path.abspath('.').replace('/Evaluation_2/monopoly_simulator','')
# upper_path = os.path.abspath('.').replace('/Evaluation_2','')
upper_path_eva = upper_path + '/Evaluation_2/monopoly_simulator'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation_2')
sys.path.append(upper_path_eva)
print('upper_path', upper_path, upper_path_eva)
#####################################
from monopoly_simulator_background.gameplay_tf import set_up_board
from monopoly_simulator.agent import Agent
from multiprocessing.connection import Client
import A2C_agent.RL_agent_v1 as RL_agent_v1
import background_agent_v3
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
from A2C_agent.logging_info import log_file_create
import datetime

from agent import Agent
import socket
import json
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.client_agent_serial')


# def make_pre_roll_move(serial_dict_to_client):
#     """
#     The agent is in the pre-roll phase and must decide what to do (next). This simple dummy agent skips the turn, and
#      doesn't do anything.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for pre_die_roll in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#     """
#     # print('simple agent pre roll client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     allowable_move_names = serial_dict_to_client['allowable_moves']
#     code = serial_dict_to_client['code']
#
#     return_to_server_dict = dict()
#     if "skip_turn" in allowable_move_names:
#         return_to_server_dict['function'] = "skip_turn"
#         return_to_server_dict['param_dict'] = dict()
#         return return_to_server_dict
#     else:
#         logger.error("Exception")
#
#
# def make_out_of_turn_move(serial_dict_to_client):
#     """
#     The agent is in the out-of-turn phase and must decide what to do (next). This simple dummy agent skips the turn, and
#      doesn't do anything.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for out_of_turn in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#     """
#     # print('simple agent oot client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     allowable_move_names = serial_dict_to_client['allowable_moves']
#     code = serial_dict_to_client['code']
#
#     return_to_server_dict = dict()
#     if "skip_turn" in allowable_move_names:
#         return_to_server_dict['function'] = "skip_turn"
#         return_to_server_dict['param_dict'] = dict()
#         return return_to_server_dict
#     else:
#         logger.error("Exception")
#
#
# def make_post_roll_move(serial_dict_to_client):
#     """
#     The agent is in the post-roll phase and must decide what to do (next). This simple dummy agent buys the property if it
#     can afford it, otherwise it skips the turn. If we do buy the property, we end the phase by concluding the turn.
#
#     Note that if your agent decides not to buy the property before concluding the turn, the property will move to
#     auction before your turn formally concludes.
#
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for post-die-roll in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#         """
#     # print('simple agent post roll client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     allowable_move_names = serial_dict_to_client['allowable_moves']
#     code = serial_dict_to_client['code']
#
#     player = current_gameboard['players'][player_name]   #respective player dictionary
#     player_current_position = player['current_position']
#     player_current_position_name = current_gameboard['location_sequence'][player_current_position]
#     current_location = current_gameboard['locations'][player_current_position_name]
#
#     return_to_server_dict = dict()
#     if "buy_property" in allowable_move_names and current_location['price'] < player['current_cash']:
#         logger.debug(player['player_name']+': We will attempt to buy '+player_current_position_name+' from the bank.')
#         if code == -1:
#             logger.debug('Did not succeed the last time. Concluding actions...')
#             return_to_server_dict['function'] = "concluded_actions"
#             return_to_server_dict['param_dict'] = dict()
#             return return_to_server_dict
#         params = dict()
#         params['player'] = player_name
#         params['asset'] = player_current_position_name
#         params['current_gameboard'] = "current_gameboard"
#         return_to_server_dict['function'] = "buy_property"
#         return_to_server_dict['param_dict'] = params
#         return return_to_server_dict
#
#     elif "concluded_actions" in allowable_move_names:
#         return_to_server_dict['function'] = "concluded_actions"
#         return_to_server_dict['param_dict'] = dict()
#         return return_to_server_dict
#
#     else:
#         logger.error("Exception")
#
#
# def make_buy_property_decision(serial_dict_to_client):
#     """
#     The decision to be made when the player lands on a location representing a purchaseable asset that is currently
#     owned by the bank. The dummy agent here returns True only if its current cash reserves are not less than the
#     asset's current price. A more sophisticated agent would consider other features in current_gameboard, including
#     whether it would be able to complete the color-set by purchasing the asset etc.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :return: A Boolean. If True, then you decided to purchase asset from the bank, otherwise False. We allow you to
#     purchase the asset even if you don't have enough cash; however, if you do you will end up with a negative
#     cash balance and will have to handle that if you don't want to lose the game at the end of your move (see notes
#     in handle_negative_cash_balance)
#     """
#     # print('simple agent buy client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     asset_name = serial_dict_to_client['asset']
#
#     player = current_gameboard['players'][player_name]
#     asset = current_gameboard['locations'][asset_name]
#
#     decision = False
#     if player['current_cash'] >= asset['price']:
#         decision = True
#     return decision
#
#
# def make_bid(serial_dict_to_client):
#     """
#     Decide the amount you wish to bid for asset in auction, given the current_bid that is currently going. If you don't
#     return a bid that is strictly higher than current_bid you will be removed from the auction and won't be able to
#     bid anymore. Note that it is not necessary that you are actually on the location on the board representing asset, since
#     you will be invited to the auction automatically once a player who lands on a bank-owned asset rejects buying that asset
#     (this could be you or anyone else).
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param asset: An purchaseable instance of Location (i.e. real estate, utility or railroad)
#     :param current_bid: The current bid that is going in the auction. If you don't bid higher than this amount, the bank
#     will remove you from the auction proceedings. You could also always return 0 to voluntarily exit the auction.
#     :return: An integer that indicates what you wish to bid for asset
#     """
#     # print('simple agent bid client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     asset_name = serial_dict_to_client['asset']
#     current_bid = serial_dict_to_client['current_bid']
#
#     asset = current_gameboard['locations'][asset_name]
#     player = current_gameboard['players'][player_name]
#
#     if current_bid < asset['price']:
#         new_bid = current_bid + (asset['price']-current_bid)/2
#         if new_bid < player['current_cash']:
#             return new_bid
#         else:   # We are aware that this can be simplified with a simple return 0 statement at the end. However in the final baseline agent
#                 # the return 0's would be replaced with more sophisticated rules. Think of them as placeholders.
#             return 0 # this will lead to a rejection of the bid downstream automatically
#     else:
#         return 0 # this agent never bids more than the price of the asset
#
#
# def handle_negative_cash_balance(serial_dict_to_client):
#     """
#     You have a negative cash balance at the end of your move (i.e. your post-roll phase is over) and you must handle
#     this issue before we move to the next player's pre-roll. If you do not succeed in restoring your cash balance to
#     0 or positive, bankruptcy proceeds will begin and you will lost the game.
#
#     The dummy agent in this case just decides to go bankrupt by returning -1. A more sophisticated agent would try to
#     do things like selling houses and hotels, properties etc. You must invoke all of these functions yourself since
#     we want to give you maximum flexibility when you are in this situation. Once done, return 1 if you believe you
#     succeeded (see the :return description for a caveat on this)
#
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :return: -1 if you do not try to address your negative cash balance, or 1 if you tried and believed you succeeded.
#     Note that even if you do return 1, we will check to see whether you have non-negative cash balance. The rule of thumb
#     is to return 1 as long as you 'try', or -1 if you don't try (in which case you will be declared bankrupt and lose the game)
#     """
#     # print('simple agent handle neg cash client')
#     player_name = serial_dict_to_client['player']
#     current_gameboard = serial_dict_to_client['current_gameboard']
#     return_to_server_dict = dict()
#     return_to_server_dict['function'] = None
#     return_to_server_dict['param_dict'] = dict()
#     return_to_server_dict['param_dict']['code'] = -1
#     return return_to_server_dict
#
#
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
#
#
# def _build_decision_agent_methods_dict():
#     """
#     This function builds the decision agent methods dictionary.
#     :return: The decision agent dict. Keys should be exactly as stated in this example, but the functions can be anything
#     as long as you use/expect the exact function signatures we have indicated in this document.
#     """
#     ans = dict()
#     ans['handle_negative_cash_balance'] = handle_negative_cash_balance
#     ans['make_pre_roll_move'] = make_pre_roll_move
#     ans['make_out_of_turn_move'] = make_out_of_turn_move
#     ans['make_post_roll_move'] = make_post_roll_move
#     ans['make_buy_property_decision'] = make_buy_property_decision
#     ans['make_bid'] = make_bid
#     ans['type'] = "decision_agent_methods"
#     return ans
#
#
# decision_agent_methods = _build_decision_agent_methods_dict() # this is the main data structure that is needed by gameplay


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
        self.upper_path = os.path.abspath('.').replace('/Evaluation_2/monopoly_simulator', '')
        self.upper_path_eva = self.upper_path + '/Evaluation_2/monopoly_simulator'
        self.game_num = 0
        self.seed = random.randint(0, 10000)
        self.retrain_signal = False

        # Read the config
        self.config_file = self.upper_path_eva + '/A2C_agent/config.ini'
        self.config_data = ConfigParser()
        self.config_data.read(self.config_file)
        self.hyperparams = self.params_read(self.config_data, 'server')
        self.rule_change_name = self.hyperparams['rule_change_name']
        self.log_file_name = self.hyperparams['log_file_name']

        #kg
        ###set gane board###
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
        self.matrix_name = 'monopoly_simulator/A2C_agent/matrix/matrix.npy'
        self.entity_name = 'monopoly_simulator/A2C_agent/matrix/entity.json'
        self.kg = None

        # Save path
        self.folder_path = None

        # A2C model parameters
        self.state_num = 104
        self.device = torch.device('cpu')
        model_path = self.upper_path_eva + '/A2C_agent/0_0_v_gat_part_seed_01200.pkl'
        self.model = torch.load(model_path)
        self.logger = logger
        self.adj_path = self.upper_path_eva + '/A2C_agent/kg_matrix_no.npy'  # TODO, new path after novelty
        self.adj = None

        self._agent_memory = dict()

        # game board read
        self.die_count = 2
        self.die_state = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
        self.cards = dict()


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
            print('Novelty Detected as ', str(self.kg_change))

        # Only when never retrained and detect novelty, will call retrain
        if self.kg_change_wait > 2 and self.kg_change_bool == False:
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

    def novelty_board_detect(self, current_gameboard):
        self.die_count = (current_gameboard['die_sequence'][0])

    def load_adj(self):
        """
        load the relationship matrix
        :param path:
        :return:
        """
        if os.path.exists(self.adj_path):
            self.adj = np.load(self.adj_path)
            adj_return = np.zeros((self.adj.shape[0], self.adj.shape[0]))
            for i in range(self.adj.shape[1] // self.adj.shape[0]):
                adj_return += self.adj[:,self.adj.shape[0] * i:self.adj.shape[0] * (i+1)]
            self.adj = adj_return

        else:
            self.adj = np.zeros((92,92))

    def params_read(self, config_data, key_word):
        params = {}
        for key in config_data[key_word]:
            v = eval(config_data[key_word][key])
            params[key] = v
        return params

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

        # print('self.folder_path', self.folder_path)

    def make_post_roll_move_agent(self, serial_dict_to_client):
        player_name = serial_dict_to_client['player']
        current_gameboard = serial_dict_to_client['current_gameboard']
        player = current_gameboard['players'][player_name]
        allowable_moves = serial_dict_to_client['allowable_moves']
        code = serial_dict_to_client['code']
        return_to_server_dict = dict()

        self.gameboard = copy.deepcopy(current_gameboard)  #TODO

        # self.interface.set_board(current_gameboard)
        # a.save_history(current_gameboard, save_path='/media/becky/Evaluation/GNOME-p3/monopoly_simulator/loc_history.pickle')

        # if player.assets == None:
        # a.save_bool(0)

        # TODO
        self.interface.get_logging_info_once(current_gameboard, self.folder_path+self.log_file_name)

        # TODO
        s = self.interface.board_to_state(current_gameboard)

        # if self.state_num != len(s):
        #     ini_current_gameboard = self.initialize_gameboard(args[1])
        #     model_retrained_path = self.retrain_from_scratch(ini_current_gameboard)
        #     self.model = torch.load(model_retrained_path)
        #     self.state_num = len(self.interface.board_to_state(args[1]))  # reset the state_num size
        #     self.kg_change_bool = True
        #     self.kg_change_wait = 0

        s = s.reshape(1, -1)
        # print('s', s)
        s = torch.tensor(s, device=self.device).float()
        self.load_adj()
        s = self.model.forward(s, self.adj)
        prob = self.model.actor(s)
        action = Categorical(prob).sample().cpu().numpy()
        action = action[0]
        actions_vector = self.interface.action_num2vec(action)
        move_actions = self.interface.vector_to_actions(current_gameboard, player, actions_vector)
        #####################

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
                return_to_server_dict['function'] = "buy_property"
                return_to_server_dict['param_dict'] = params
                return return_to_server_dict

        # improve property
        if action_choices.improve_property in move_action:  # beef up full color sets to maximize rent potential.
            if "improve_property" in allowable_moves:
                params = dict()
                params['player'] = "player_1"
                params['asset'] = space_action[0]
                params['current_gameboard'] = "current_gameboard"
                return_to_server_dict['function'] = "improve_property"
                return_to_server_dict['param_dict'] = params
                return return_to_server_dict

        # free-mortgage
        if action_choices.free_mortgage in move_action:
            if "free_mortgage" in allowable_moves:
                # free mortgages till we can afford it. the second condition should not be necessary but just in case.
                params = dict()
                params['player'] = "player_1"
                params['asset'] = space_action[0]
                params['current_gameboard'] = "current_gameboard"
                return_to_server_dict['function'] = "free_mortgage"
                return_to_server_dict['param_dict'] = params
                return return_to_server_dict

        # mortgage
        if action_choices.mortgage_property in move_action:
            if "mortgage_property" in allowable_moves:
                params = dict()
                params['player'] = "player_1"
                params['asset'] = space_action[0]
                params['current_gameboard'] = "current_gameboard"
                return_to_server_dict['function'] = "mortgage_property"
                return_to_server_dict['param_dict'] = params
                return return_to_server_dict
        return_to_server_dict['function'] = "concluded_actions"
        return_to_server_dict['param_dict'] = dict()
        return return_to_server_dict

    def initialize_gameboard(self, gameboard):
        """
        Intialize the used gameboard
        :param gameboard:
        :return:
        """
        self.schema_path = self.folder_path + '/schema.json'

        game_schema = json.load(open(self.upper_path + '/Evaluation_2/monopoly_game_schema_v1-2.json', 'r'))
        print('game_schema', game_schema.keys())
        # print('locations', game_schema['locations']['location_states'][1])
        # print('schema', game_schema['location_sequence'])
        # print('gameboard', gameboard['locations']['Mediterranean Avenue'])

        # location
        game_schema['locations']['location_count'] = len(gameboard['location_sequence'])
        game_schema['location_sequence'] = gameboard['location_sequence']
        location_states = []
        for name in game_schema['location_sequence']:
            location_states.append(gameboard['locations'][name])
            if name == 'Go':
                game_schema['go_position'] = int(gameboard['locations'][name]['start_position'])
        game_schema['locations']['location_states'] = location_states

        # go_increment TODO

        # die TODO add novelty from kg
        game_schema['die']['die_count'] = self.die_count
        game_schema['die']['die_state'] = self.die_state

        #cards
        community_chest_list = []
        for name in self.cards['community_chest']:
            community_chest_list.append(self.cards['community_chest'][name])
        print(community_chest_list)
        print('===')
        print('cards', game_schema['cards']['community_chest']['card_states'])
        # print('cards', game_schema['cards']['chance'])







        set_up_board(self.upper_path + '/Evaluation_2/monopoly_game_schema_v1-2.json',
                     self.player_decision_agents,
                     self.num_players)
        return ini_gameboard

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
        while True:
            data_from_server = self.conn.recv(100000)
            data_from_server = data_from_server.decode("utf-8")

            data_dict_from_server = json.loads(data_from_server)
            func_name = data_dict_from_server['function']
            if 'shut' in func_name:
                print('func_name', self.game_num, func_name)

            # When the tournament begins, we need to
            if func_name == "start_tournament":
                self.define_path(data_dict_from_server['path'])
                self.logger.info('Tournament starts!')
                result = 1

            # Before simulating each game, we have to make sure if we need retrain the network
            elif func_name == "startup": # args = dict() with 3 keys :'current_gameboard', 'indicator', 'function'
                # Clear interface history and set the init for interface
                self.interface.clear_history(self.folder_path+self.log_file_name)
                self.interface.set_board(data_dict_from_server['current_gameboard'])  # args[0] is current_gameboard
                s = self.interface.board_to_state(data_dict_from_server['current_gameboard'])
                # print('gameboard, startup', data_dict_from_server['current_gameboard']['cards'].keys())
                self.game_num += 1  # number of games simulated
                self.logger.info(str(self.game_num) + ' th game starts!')

                # We need to have a baseline of the gameboard
                if self.game_num == 1:
                    self.gameboard_ini = data_dict_from_server['current_gameboard']
                    self.kg = KG_OpenIE_eva(self.gameboard_ini,
                                            self.matrix_name,
                                            self.entity_name,
                                            config_file=self.config_file)

                # # Detect card type and number change with comparing gameboard
                # if self.novelty_card == None:
                #     self.novelty_card = detect_card_nevelty(data_dict_from_server['current_gameboard'], self.gameboard_ini)
                # if self.novelty_bank == None:
                #     self.novelty_bank = detect_contingent(data_dict_from_server['current_gameboard'], self.gameboard_ini)
                #
                # if self.novelty_card:
                #     self.kg_change.append(self.novelty_card)
                #     self.novelty_card = []
                # if self.novelty_bank:
                #     self.kg_change.append(self.novelty_bank)
                #     self.novelty_bank = []

                # if board size changed, before the game, we need to retrain the NN
                if self.state_num != len(s) or self.retrain_signal:
                    ini_current_gameboard = self.initialize_gameboard(data_dict_from_server['current_gameboard'])  #TODO
                    model_retrained_path, adj_path = self.retrain_from_scratch(ini_current_gameboard)

                    self.model = torch.load(model_retrained_path)
                    self.state_num = len(self.interface.board_to_state(data_dict_from_server['current_gameboard']))  # reset the state_num size
                    self.kg_change_bool = True
                    self.retrain_signal = False
                    self.kg_change_wait = 0
                    self.adj_path = adj_path

                # result = getattr(self, func_name)(*args)


            # When each game ends, we run the KG, but we don not shutdown the connection
            elif func_name == 'shutdown':
                serial_dict_to_client = data_dict_from_server
                self.kg_run(self.gameboard)
                result = shutdown(serial_dict_to_client, self)
                self.logger.info(str(self.game_num) + ' th game stops!')

            # When calling agent to make decision
            elif func_name == 'make_post_roll_move':
                # print('gameboard post', data_dict_from_server['current_gameboard']['cards'].keys())
                serial_dict_to_client = data_dict_from_server
                result = self.make_post_roll_move_agent(serial_dict_to_client)

            elif func_name == 'make_out_of_turn_move':
                # print('gameboard out', data_dict_from_server['current_gameboard']['cards'].keys())
                serial_dict_to_client = data_dict_from_server
                self.gameboard = serial_dict_to_client['current_gameboard']
                server_result, self._agent_memory = self.make_out_of_turn_move(serial_dict_to_client, self._agent_memory)
                serial_dict_to_server = dict()
                serial_dict_to_server['function'] = server_result[0]
                serial_dict_to_server['param_dict'] = server_result[1]
                result = serial_dict_to_server
                # print('result', result)

            elif func_name == 'make_pre_roll_move':
                #update cards TODO check novelty
                self.cards['chance'] = data_dict_from_server['current_gameboard']['cards']['picked_chance_card_details']
                self.cards['community_chest'] = data_dict_from_server['current_gameboard']['cards']['picked_community_chest_card_details']

                serial_dict_to_client = data_dict_from_server
                result,self._agent_memory = self.make_pre_roll_move(serial_dict_to_client, self._agent_memory)

            elif func_name == 'make_bid':
                serial_dict_to_client = data_dict_from_server
                result = self.make_bid(serial_dict_to_client)

            elif func_name == 'make_buy_property_decision':
                serial_dict_to_client = data_dict_from_server
                result = self.make_buy_property_decision(serial_dict_to_client)

            elif func_name == 'handle_negative_cash_balance':
                serial_dict_to_client = data_dict_from_server
                server_result, self._agent_memory = self.handle_negative_cash_balance(serial_dict_to_client, self._agent_memory)
                serial_dict_to_server = dict()
                serial_dict_to_server['function'] = server_result[0]
                serial_dict_to_server['param_dict'] = server_result[1]
                result = serial_dict_to_server

            # Send we will close the connection now back to server
            elif func_name == "end_tournament":
                result = 1
                self.logger.info('Tournament Finished!')

            else:
                serial_dict_to_client = data_dict_from_server
                result = getattr(self, func_name)(serial_dict_to_client)

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

