import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
####################
from monopoly_simulator import initialize_game_elements
import numpy as np
from monopoly_simulator.card_utility_actions import move_player_after_die_roll
from monopoly_simulator_background import simple_background_agent_becky_v1
import json
from monopoly_simulator import diagnostics
from monopoly_simulator.action_choices import *
from monopoly_simulator import location
from monopoly_simulator_background.agent_helper_functions import identify_free_mortgage
import logging

from monopoly_simulator_background.log_setting import ini_log_level, set_log_level
logger = set_log_level()

class Interface(object):

    def __init__(self):
        self.board_owned = []
        self.board_building = []
        self.board_state = []
        self.state_space = []
        self.masked_actions = []
        self.move_actions = []
        self.action_space_num = 1 + 22 + 28 + 28 + 1
        self.action_space = []
        self.site_space = []
        self.loc_history = set()
        self.num_players = 0

    def mapping(self, name):
        name = name.replace('&','-')
        return '-'.join(name.split(' '))

    def clear_history(self, file_path=upper_path + '/KG_rule/game_log.txt'):
        file = open(file_path, "w")
        file.write('Game start! \n')
        file.close()

        self.loc_history.clear()

    def save_history(self, game_elements):
        for p in game_elements['players']:
            if p.current_position:
                self.loc_history.add(p.current_position)

    def get_logging_info(self, game_elements, current_player_index, file_path=upper_path+'/KG_rule/game_log.txt'):
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
        # Set the full board
        self.board_state = [i.name for i in gameboard['location_sequence']]

        # Set the property which can build property
        board_building = []
        for k, v in gameboard['location_objects'].items():
            if type(v) ==  location.RealEstateLocation:
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
        state_space = []
        # Ownership of property => # of board states
        # -1 means other players, 0 means bank, and 1 means agent/ player 1
        state_space_owned = []
        for space in self.board_state:
            if type(current_board['location_objects'][space]) != location.DoNothingLocation and \
                    type(current_board['location_objects'][space]) != location.ActionLocation and \
                    type(current_board['location_objects'][space]) != location.TaxLocation:
                if current_board['location_objects'][space].owned_by == current_board['bank']:
                    state_space_owned.append(0)
                elif current_board['location_objects'][space].owned_by.player_name == 'player_1':
                    state_space_owned.append(1)
                else:
                    state_space_owned.append(-1)
            else:
                state_space_owned.append(0)
        state_space += state_space_owned

        # Number of house in the property => # of houses in the space: 0,1,2,3,4,5
        state_space_building = []
        for space in self.board_state:
            if type(current_board['location_objects'][space]) == location.RealEstateLocation:
                if current_board['location_objects'][space].num_hotels == 0:
                    state_space_building.append(current_board['location_objects'][space].num_houses)
                else:
                    state_space_building.append(5) # 5 denotes hotel
            else:
                state_space_building.append(0)
        state_space += state_space_building

        # Position => n positions of players n = # of players
        sorted_player = sorted(current_board['players'], key=lambda player: int(player.player_name[-1]))
        state_space_position = [p.current_position for p in sorted_player]
        for i, pos in enumerate(state_space_position):
            state_space_position[i] = int(pos) if pos else 0

        state_space += state_space_position

        # Card Ownership
        #2 # of get-out_of_jail_card of players
        state_space_card = []
        com_card = [p.has_get_out_of_jail_community_chest_card for p in sorted_player]
        chance_card = [p.has_get_out_of_jail_chance_card for p in sorted_player]
        # first num denotes the # of card agent has
        num_card_agent = int(com_card[0] + chance_card[0])
        state_space_card.append(num_card_agent)
        # second num denotes the cards other players have
        num_card_others = sum(com_card) + sum(chance_card) - num_card_agent
        state_space_card.append(num_card_others)
        state_space += state_space_card

        # Cash
        # n cash ratio for all the players n = # of players

        state_space_cash = [int(p.current_cash) / 1000 for p in sorted_player]
        # print('cash',state_space_cash)
        state_space += state_space_cash

        np.set_printoptions(suppress=True)
        self.state_space = np.array(state_space)

        return self.state_space

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
            print('current_player.current_position',current_player.current_position)
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
            if state_space[index_space] != 0:
                return True
            if index_space in state_space[-2 * self.num_players - 2 : -self.num_players - 2]:
                return True

        return False









