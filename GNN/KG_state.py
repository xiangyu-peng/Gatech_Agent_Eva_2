import sys
import os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
from KG_rule.openie_triple import KG_OpenIE

class KGState():

    def __init__(self, gameboard):
        self.rel_matrix = None  # output sparse matrix of state
        self.nodes = []
        self.relations_matrix = ['locate', 'own', 'house', 'cash']
        self.board_name = self.set_gameboard(gameboard)
        self.sparse_matrix_dict = self.generate_empty_matrix()

    def set_gameboard(self, gameboard):
        """
        set boardname before each game
        :param gameboard: dict()
        :return: board names e.g. a list of 40 names
        """
        board_name = [i.name for i in gameboard['location_sequence']]
        for i, name in enumerate(board_name):
            board_name[i] = '-'.join(name.split(' '))
        return board_name

    def generate_empty_matrix(self):
        sparse_matrix_dict = dict()
        sparse_matrix_dict['number_nodes'] = dict()
        sparse_matrix_dict['nodes_number'] = dict()
        sparse_matrix_dict['out'] = dict()
        # sparse_matrix_dict['in'] = dict()
        sparse_matrix_dict['number_rel'] = dict()
        sparse_matrix_dict['rel_number'] = dict()
        sparse_matrix_dict['all_rel'] = dict()

        # Define the relation/edge number
        for i, rel in enumerate(self.relations_matrix):
            sparse_matrix_dict['number_rel'][i] = rel
            sparse_matrix_dict['rel_number'][rel] = i

        # Define the number of nodes
        # name of location
        sparse_matrix_dict['number_nodes'][0] = 'player'
        sparse_matrix_dict['number_nodes'][1] = 'player_1'
        sparse_matrix_dict['number_nodes'][2] = 'player_2'

        for i, node in enumerate(self.board_name):
            sparse_matrix_dict['number_nodes'][i+3] = node
            if node in sparse_matrix_dict['nodes_number']:
                id_value = sparse_matrix_dict['nodes_number'][node]
                if type(id_value) == list:
                    id_value.append(i + 3)
                else:
                    id_value = [id_value] + [i + 3]
                sparse_matrix_dict['nodes_number'][node] = id_value
            else:
                sparse_matrix_dict['nodes_number'][node] = i + 3

        # House
        # 0/1/2/3/4/5
        for j in range(6):
            sparse_matrix_dict['number_nodes'][i + j + 4] = 'House_' + str(j)
            sparse_matrix_dict['nodes_number']['House_' + str(j)] = i + j + 4

        # cash
        for k in range(6):
            sparse_matrix_dict['number_nodes'][i + j + k + 5] = 'Cash_' + str(k)
            sparse_matrix_dict['nodes_number']['Cash_' + str(k)] = i + j + k + 5

        self.node_number = len(sparse_matrix_dict['number_nodes'].keys())

        # # Define 'in' column names
        # for rel in self.relations_matrix:
        #     sparse_matrix_dict['in'][rel] = dict()
        #     for node in range(self.node_number):
        #         sparse_matrix_dict['in'][rel][node] = 0

        # Define 'out' column names\
        sparse_matrix_dict = self.init_dict(sparse_matrix_dict)

        return sparse_matrix_dict

    def init_dict(self, sparse_matrix_dict):

        for rel in self.relations_matrix:
            sparse_matrix_dict['out'][rel] = dict()
            for node in range(self.node_number):
                sparse_matrix_dict['out'][rel][node] = None
        sparse_matrix_dict['out']['own'][1] = []
        sparse_matrix_dict['out']['own'][2] = []

        return sparse_matrix_dict

    def dict_to_matrix(self):
        """
        state vector to matrix
        :param state: vector - list
        :return: a dict
        """
        matrix = None
        for node_number in range(len(self.sparse_matrix_dict['number_nodes'].keys())):
            # node_name = self.sparse_matrix_dict['nodes_number'][node_number]
            line_vector_number = []
            for i, rel in enumerate(self.relations_matrix):
                if self.sparse_matrix_dict['out'][rel][node_number]:
                    # print(node_number, rel, self.sparse_matrix_dict['out'][rel][node_number])
                    if type(self.sparse_matrix_dict['out'][rel][node_number]) == list:
                        line_vector_number += self.sparse_matrix_dict['out'][rel][node_number]
                    else:
                        line_vector_number.append(int(self.sparse_matrix_dict['out'][rel][node_number]))
            line_vector = [0 if i not in line_vector_number else 1 for i in range(len(self.sparse_matrix_dict['number_nodes'].keys()))]
            line_vector = np.array([line_vector])
            if node_number:
                matrix = np.concatenate((matrix, line_vector), axis=0)
            else:
                matrix = line_vector
        return matrix


from monopoly_simulator import initialize_game_elements
import numpy as np
from monopoly_simulator.card_utility_actions import move_player_after_die_roll
from monopoly_simulator_background import simple_background_agent_becky_v1
import json
from monopoly_simulator import diagnostics
from monopoly_simulator.action_choices import *
from monopoly_simulator import location
from monopoly_simulator_background.agent_helper_functions import identify_free_mortgage

class Interface(object):

    def __init__(self):
        self.board_owned = []
        self.board_building = []
        self.board_state = []
        self.state_space = []
        self.masked_actions = []
        self.move_actions = []
        self.action_space_num = 2
        self.action_space = []
        self.site_space = []
        self.loc_history = set()
        self.num_players = 0
        self.upper_path = '/media/becky/GNOME-p3'

        self.kg_state = None

    def mapping(self, name):
        name = name.replace('&','-')
        return '-'.join(name.split(' '))

    def clear_history(self, file_path=None):  # file_path is full path
        file = open(file_path, "w")
        file.write('Game start! \n')
        file.close()

        self.loc_history.clear()

    def save_history(self, game_elements):
        for p in game_elements['players']:
            if p.current_position:
                self.loc_history.add(p.current_position)

    def get_logging_info(self, game_elements, current_player_index, file_path=None):
        file = open(file_path, "w")
        current_player = game_elements['players'][current_player_index]

        # Record history of dice after one game
        # if current_player.current_cash <= 0 or num_active_players == 1:
        for i in game_elements['history']['return']:
            if type(i) == list:
                file.write('Dice => '+ str(i) + '\n')

        #Record card class
        for i in game_elements['history']['param']:
            if 'card' in i:
                if 'pack' in i:
                    file.write('C' + str(i['pack'])[1:] +' Card => ' + i['card'].name + '\n')
                    # print(i['pack'])
                else:
                    if i['card'] in i['current_gameboard']['community_chest_cards']:
                        file.write('Community_chest' + ' Card => ' + i['card'].name + '\n')
                    else:
                        file.write('Chance' + ' Card => ' + i['card'].name + '\n')

        for i, his in enumerate(game_elements['history']['param']):
            if 'card' in his:
                if 'pack' in his:
                    file.write('C' + str(his['pack'])[1:] +' Card => ' + his['card'].name + '\n')
                    # print(i['pack'])
                else:
                    if his['card'] in his['current_gameboard']['community_chest_cards']:
                        file.write('Community_chest' + ' Card => ' + his['card'].name + '\n')
                    else:
                        file.write('Chance' + ' Card => ' + his['card'].name + '\n')

                file.write(his['card'].name + ' is classified as ' + his['card'].card_type  + '\n')

                # if 'amount' in game_elements['history']['param'][i-1]:
                #     if 'go-to-nearest' not in his['card'].name:
                #         file.write(his['card'].name + ' is cost at ' + str(game_elements['history']['param'][i-1]['amount']) + '\n')

        # Add to kg
        for loc in self.loc_history:
            space = game_elements['location_sequence'][int(loc)]
            file.write(self.mapping(space.name) + ' is located at ' + str(loc) + '\n')
            loc_class = space.loc_class
            file.write(self.mapping(space.name) + ' is classified as ' + str(space.loc_class) + '\n')

            if loc_class == 'real_estate':
                file.write(self.mapping(space.name) + ' is colored as ' + str(space.color) + '\n')
                file.write(self.mapping(space.name) + ' is price-1-house at ' + str(space.price_per_house) + '\n')
                file.write(self.mapping(space.name) + ' is rented-1-hotel at ' + str(space.rent_hotel) + '\n')
                file.write(self.mapping(space.name) + ' is rented-0-house at ' + str(space.rent) + '\n')
                file.write(self.mapping(space.name) + ' is rented-1-house at ' + str(space.rent_1_house) + '\n')
                file.write(self.mapping(space.name) + ' is rented-2-house at ' + str(space.rent_2_houses) + '\n')
                file.write(self.mapping(space.name) + ' is rented-3-house at ' + str(space.rent_3_houses) + '\n')
                file.write(self.mapping(space.name) + ' is rented-4-house at ' + str(space.rent_4_houses) + '\n')

            if loc_class in ['real_estate', 'railroad',  'utility']:
                file.write(self.mapping(space.name) + ' is mortgaged at ' + str(space.mortgage) + '\n')
                file.write(self.mapping(space.name) + ' is priced at ' + str(space.price) + '\n')

            if loc_class == 'tax':
                file.write(self.mapping(space.name) + ' is cost at ' + str(space.amount_due) + '\n')

        # Add go_increment to the file
        file.write('GO is incremtent as ' + str(game_elements['go_increment']) + '\n')
        file.write('GO is located at ' + str(game_elements['go_position']) + '\n')

        file.close()

    def set_board(self, gameboard):
        self.kg_state = KGState(gameboard)
        if isinstance(gameboard, str):
            gameboard_path = gameboard
            with open(self.upper_path+gameboard_path, 'r') as load_f:
                gameboard_load = json.load(load_f)
                self.board_state = gameboard_load['locations']['location_sequence']

                board_building, board_owned = [], []
                for loc in gameboard_load['locations']['location_objects']:
                    if loc["loc_class"] == "real_estate":
                        board_building.append(loc["name"])
                    if loc["loc_class"] != "do_nothing" and loc["loc_class"] != "action"\
                            and loc["loc_class"] != "tax":
                        board_owned.append(loc["name"])
                self.board_building, self.board_owned = board_building, board_owned

                self.num_players = len(gameboard_load['players'])

        else:
            # Set the full board
            self.board_state = [i.name for i in gameboard['location_sequence']]

            # Set the property which can build property
            board_building = []
            for k, v in gameboard['location_objects'].items():
                if type(v) == location.RealEstateLocation:
                    board_building.append(k)
            self.board_building = board_building

            # Set the property can be owned
            board_owned = []
            for k, v in gameboard['location_objects'].items():
                if type(v) !=  location.DoNothingLocation and type(v) !=  location.ActionLocation and type(v) !=  location.TaxLocation:
                    board_owned.append(k)
            self.board_owned = board_owned

            #Set the players number
            self.num_players = len(gameboard['players'])

    #state_space = 28+22+n+n+2 = 56
    def board_to_state(self, current_board):
        """
        Transfer the game board into a vector
        :param current_board: a dict, current game board
        :return: state_space: a numpy array
        """
        self.kg_state.sparse_matrix_dict = self.kg_state.init_dict(self.kg_state.sparse_matrix_dict)  # empty the dict

        if isinstance(current_board, str):
            return [0 for i in range(2 * len(self.board_state) + 2 + 2 * self.num_players)]

        # player location:
        sorted_player = sorted(current_board['players'], key=lambda player: int(player.player_name[-1]))
        if sorted_player[0].current_position:
            self.kg_state.sparse_matrix_dict['out']['locate'][0] = int(sorted_player[0].current_position + 3)
        else:
            self.kg_state.sparse_matrix_dict['out']['locate'][0] = 3
        # own
        state_space = []
        # Ownership of property => # of board states
        # -1 means other players, 0 means bank, and 1 means agent/ player 1
        for space in self.board_state:
            if type(current_board['location_objects'][space]) != location.DoNothingLocation and \
                    type(current_board['location_objects'][space]) != location.ActionLocation and \
                    type(current_board['location_objects'][space]) != location.TaxLocation:
                space_number = self.kg_state.sparse_matrix_dict['nodes_number']['-'.join(space.split(' '))]
                if current_board['location_objects'][space].owned_by == current_board['bank']:
                    pass
                elif current_board['location_objects'][space].owned_by.player_name == 'player_1':
                    self.kg_state.sparse_matrix_dict['out']['own'][1].append(space_number)
                elif current_board['location_objects'][space].owned_by.player_name == 'player_2':
                    self.kg_state.sparse_matrix_dict['out']['own'][2].append(space_number)

        # Number of house in the property => # of houses in the space: 0,1,2,3,4,5
        for space in self.board_state:
            if type(current_board['location_objects'][space]) == location.RealEstateLocation:
                space_number = self.kg_state.sparse_matrix_dict['nodes_number']['-'.join(space.split(' '))]
                if current_board['location_objects'][space].num_hotels == 0:
                    # self.kg_state.sparse_matrix_dict['out']['house'][space_number] = \
                    #     state_space_num + current_board['location_objects'][space].num_houses
                    house_number = self.kg_state.sparse_matrix_dict['nodes_number']['House_' + str(current_board['location_objects'][space].num_houses)]
                    self.kg_state.sparse_matrix_dict['out']['house'][space_number] = house_number
                else:
                    house_number = self.kg_state.sparse_matrix_dict['nodes_number']['House_5']
                    self.kg_state.sparse_matrix_dict['out']['house'][space_number] = house_number

        # Cash
        # n cash ratio for all the players n = # of players
        for i, p in enumerate(sorted_player):
            if p.current_cash // 500 < 5:
                cash_number = self.kg_state.sparse_matrix_dict['nodes_number']['Cash_' + str(max(0,int(p.current_cash // 500)))]
            else:
                cash_number = self.kg_state.sparse_matrix_dict['nodes_number']['Cash_5']
            self.kg_state.sparse_matrix_dict['out']['cash'][1 + i] = cash_number
        return self.kg_state.dict_to_matrix()

    def get_masked_actions(self, allowable_actions, param, current_player):
        """
        This function is to transfer allowable actions to vector/array.
        :param allowable_actions: a set. allowed actions from player.py.
        :param param: dict. parameters for allowed actions from play.py.
        :param current_player: player class.
        :return: masked_actions, a list.
        """
        masked_actions = []
        #1 first denotes the action -> buy or not buy
        if buy_property in allowable_actions:
            masked_actions.append(1)
        else:
            masked_actions.append(0)

        # # Improve property => # of game states
        # if improve_property in allowable_actions:
        #     if param:
        #         for space in self.board_state:
        #             masked_actions.append(1) if space in param['asset'] else masked_actions.append(0)
        #     else:
        #         for space in self.board_state:
        #             masked_actions.append(0)
        # else:
        #     masked_actions += [0 for i in range(len(self.board_state))]

        # # Allowed_morgage => # of game states
        # owned_space = [asset.name for asset in current_player.assets]
        # mortgaged_assets = [asset.name for asset in current_player.mortgaged_assets]
        # for space in self.board_state:
        #     masked_actions.append(1) if space in owned_space and space not in mortgaged_assets else masked_actions.append(0)

        # # Free morgage
        # potentials = identify_free_mortgage(current_player)
        # potentials = [asset.name for asset in potentials]
        # for space in self.board_state:
        #     masked_actions.append(1) if space in potentials else masked_actions.append(0)

        # 1 action: always allowed : conclude the actions = skip = do nothing.
        masked_actions.append(1)

        self.masked_actions = masked_actions
        return self.masked_actions

    #action_# is 80
    def action_num2vec(self, action):
        '''
        Action taken from A2C agent is a interger/index, need to transform to vector.
        :param action: a integer: index of action vector.
        :return: a vector with only one 1 otherwise 0 for the whole vector
        '''
        actions_vec = [0 for i in range(self.action_space_num)]
        actions_vec[int(action)] = 1
        return actions_vec

    #action space 1+22+28+28+1 = 80
    def get_action_space(self, current_board, current_player):
        """
        :param current_board: a dict, the game board with history.
        :param current_player: player object, the player moving now: in our game, is player_1.
        :return:
        """

        if self.site_space:
            if current_player.current_position != None:
                self.site_space[0] = current_board['location_objects'][self.board_state[current_player.current_position]]
        else:

            # 1 first denotes the action -> buy or not buy
            self.action_space.append(buy_property)
            self.site_space.append(current_board['location_objects'][self.board_state[current_player.current_position]])

            # # Improve property =>  # of game states
            # for space in self.board_state:
            #     self.site_space.append(current_board['location_objects'][space])
            # for i in range(len(self.board_state)):
            #     self.action_space.append(improve_property)

            # # Mortgage =>  # of game states
            # for space in self.board_state:
            #     self.site_space.append(current_board['location_objects'][space])
            # for i in range(len(self.board_state)):
            #     self.action_space.append(mortgage_property)

            # # Free mortgage =>  # of game states
            # for space in self.board_state:
            #     self.site_space.append(current_board['location_objects'][space])
            # for i in range(len(self.board_state)):
            #     self.action_space.append(free_mortgage)

            # Do nothing => 1
            self.action_space.append(concluded_actions)
            self.site_space.append(current_board['location_objects']['Go'])

        return self.action_space, self.site_space

    #     #take into array/vector and output the actions
    # actions_vector_default = [1] + [0]

    def vector_to_actions(self, current_board, current_player, actions_vector=None):
        '''
        Take into array/vector of actions from agent and output the actions which are taken into env
        :param current_board: dict.
        :param current_player: class.
        :param actions_vector: list.
        :return: a action function.
        '''
        move_actions = []
        self.action_space, self.site_space = self.get_action_space(current_board, current_player)
        try:
            actions_vector = actions_vector.tolist()
        except AttributeError:
            print(' ')

        for i,action in enumerate(actions_vector):
            if action == 1:
                move_actions.append((self.action_space[i], self.site_space[i]))

        self.move_actions = move_actions
        return self.move_actions

    def check_relative_state(self, state_space, spaces_set):
        """
        Given the state_space, which is a list of current game state and the novelty set which involves all the
        spaces with novelty, then return if the state has novelty already.
        :param state_space: a list, the game state from board_to_state()
        :param spaces_set: a set, involves all the spaces having novelty
        :return: a boolean, denotes whether novelty has been introduced into this state already
        """
        for space in spaces_set:
            index_space = self.board_state.index(space)
            # if state_space[index_space] != 0:
            #     return True
            # if index_space in state_space[-2 * self.num_players - 2 : -self.num_players - 2]:
            if state_space[40:-12][index_space] == 1:
                return True

        return False













