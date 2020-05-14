import os, sys
upper_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append('/media/becky/GNOME-p3')
sys.path.append(upper_path)
from monopoly_simulator.agent import Agent
from multiprocessing.connection import Client
import A2C_agent.RL_agent_v1 as RL_agent_v1
from A2C_agent.novelty_detection import KG_OpenIE_eva
from A2C_agent.interface_eva import Interface_eva
import torch
from monopoly_simulator_background.vanilla_A2C import *
from monopoly_simulator import action_choices
from configparser import ConfigParser
from monopoly_simulator_background.vanilla_A2C_main_v3 import MonopolyTrainer

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

        #Read the config
        self.config_file = upper_path + '/A2C_agent/config.ini'
        self.config_data = ConfigParser()
        self.config_data.read(self.config_file)
        self.hyperparams = self.params_read(self.config_data, 'server')
        self.kg_rel_path = upper_path + self.hyperparams['kg_rel_path']
        self.rule_change_path = upper_path + self.hyperparams['rule_change_path']

        #A2C model parameters
        self.state_num = 90
        self.device = torch.device('cuda:0')
        model_path = upper_path + '/A2C_agent/weights/orginal.okl'
        self.model = torch.load(model_path)



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
        self.interface.set_board(current_gameboard)
        self.interface.get_logging_info_once(current_gameboard)
        self.interface.get_logging_info(current_gameboard)

        #######Add kg here######
        self.kg.set_gameboard(current_gameboard)
        self.kg.build_kg_file(upper_path + '/A2C_agent/log/game_log.txt', level='rel', use_hash=False)
        self.interface.clear_history()  #clear the log history in file

        # self.kg.save_file(self.kg.kg_rel, self.kg_rel_path)
        if 'is rented-0-house at' in self.kg.kg_rel.keys():
            if 'Indiana-Avenue' in self.kg.kg_rel['is rented-0-house at'].keys():
                print(self.kg.kg_rel['is rented-0-house at']['Indiana-Avenue'])

        # print(self.kg.kg_change)

        ####no need for evaluation####
        # self.kg.dict_to_matrix()
        # self.kg.save_matrix()
        # self.kg.save_vector()
        ###############################

        # Visulalize the novelty change in the network
        if self.kg.kg_change != self.kg_change[-1]:
            file = open(self.rule_change_path, "a")
            file.write(str(self.kg.kg_change) + ' \n')
            file.close()
            self.kg_change_bool = True
            self.kg_change.append(self.kg.kg_change[:])

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
        print('self.kg_change', self.kg_change)

        params = self.params_read(self.config_data, 'hyper')
        trainer = MonopolyTrainer(params=params, device_id=None, gameboard=gameboard, kg_use=False)
        trainer.train()


    def play_remote_game(self, address=('localhost', 6001), authkey=b"password"):
        """
        Connects to a ServerAgent and begins the loop of waiting for requests and responding to them.
        @param address: Tuple, the address and port number. Defaults to localhost:6000
        @param authkey: Byte string, the password used to authenticate the client. Must be same as server's authkey.
            Defaults to "password"
        """
        self.conn = Client(address, authkey=authkey)
        while True:
            func_name, args = self.conn.recv()
            print(func_name)
            if func_name == "startup":
                print('!!!!!!!!!!!!!!',func_name)

            if func_name == 'make_post_roll_move':  # args = (player, current_gameboard, allowable_moves, code)
                print('make_post_roll_move')
                if self.kg_change_bool:
                    ini_current_gameboard = self.initialize_gameboard(args[1])
                    self.retrain_from_scratch(ini_current_gameboard)
                    device = torch.device('cuda:0')
                    model_path = upper_path + '/A2C_agent/weights/push_buy_tf_ne_v3_1.pkl'  # New path of retrained model
                    self.model = torch.load(model_path)
                    self.state_num = len(self.interface.board_to_state(args[1]))  # reset the state_num size
                    self.kg_change_bool = False


                result = self.make_post_roll_move_agent(args)

            elif func_name == 'handle_negative_cash_balance':
                result = getattr(self, func_name)(*args)
                self.kg_run(args[-1])
            else:
                result = getattr(self, func_name)(*args)

            self.conn.send(result)

            if func_name == "shutdown":
                self.conn.close()
                break

    def make_post_roll_move_agent(self, args):
        player, current_gameboard, allowable_moves, code = args

        self.interface.set_board(current_gameboard)
        # a.save_history(current_gameboard, save_path='/media/becky/Evaluation/GNOME-p3/monopoly_simulator/loc_history.pickle')

        # if player.assets == None:
        # a.save_bool(0)
        self.interface.get_logging_info_once(current_gameboard)

        s = self.interface.board_to_state(current_gameboard)
        # print(s)

        if self.state_num != len(s):
            ini_current_gameboard = self.initialize_gameboard(current_gameboard)
            self.retrain_from_scratch(ini_current_gameboard)
            # device = torch.device('cpu')
            model_path = upper_path + '/A2C_agent/weights/push_buy_tf_ne_v3_1.pkl'  # New path of retrained model
            self.model = torch.load(model_path)
            self.state_num = len(s)  # reset the state_num size
            self.kg_change_bool = False

        s = s.reshape(1, -1)
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