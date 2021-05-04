import os, sys
upper_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(upper_path)

from monopoly_simulator import initialize_game_elements
import numpy as np
from monopoly_simulator.card_utility_actions import move_player_after_die_roll
import json
from monopoly_simulator import diagnostics
from monopoly_simulator.action_choices import *
from monopoly_simulator import location
import pickle

class Interface_eva(object):
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
        self.loc_history = set()  # record the history of locations each game

    def mapping(self, name):
        """
        In order to deal with OpenIE bad implementation,
        we need to map the board like this!
        :param name:
        :return:
        """
        name = name.replace('&','-')
        return '-'.join(name.split(' '))


    def clear_history(self, file_path= upper_path+'/A2C_agent_2/logs/game_log.txt'):  #, save_path=upper_path+'/A2C_agent/log/history_loc.pickle'):
        """
        Clear the history of file and loc history before running the game
        :param file_path: The file restoring logging info
        :return:
        """
        # Clear the logging info history
        file = open(file_path, "w")
        file.write('Game start! \n')
        file.close()

        # Clear the loc history
        self.loc_history.clear()


    def get_logging_info_once(self, game_elements, file_path=upper_path+'/A2C_agent_2/logs/game_log.txt'):
        """
        Write Dice and Card history to logging info
        :param game_elements:
        :param file_path:
        :return:
        """
        # Write Dice logging history to file
        file = open(file_path, "a")
        # for i in game_elements['die_sequence']:
        #     if type(i) == list:
        #         file.write('Dice => '+ str(i) + '\n')

        #Record card class to file
        if game_elements['cards']['picked_chance_cards']:
            file.write('Chance' + ' Card => ' + game_elements['cards']['picked_chance_cards'][0] + '\n')
        if game_elements['cards']['picked_community_chest_cards']:
            file.write('Community_chest' + ' Card => ' + game_elements['cards']['picked_community_chest_cards'][0] + '\n')
        file.close()

        #  Record history of locations

        # print(game_elements['players']['player_1']['current_position'])
        # for p in game_elements['players']:
        #     if p.current_position:
        self.loc_history.add(game_elements['players']['player_1']['current_position'])

        # if save_path:
        #     with open(save_path, 'rb') as f:
        #         loc_history = pickle.load(f)
        #         if loc_history:
        #             self.loc_history = loc_history
        #         else:
        #             self.loc_history = set()

        # if save_path:
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(self.loc_history, f)


    def get_logging_info(self, game_elements, file_path=upper_path+'/A2C_agent_2/logs/game_log.txt'):
        """
        Write the info of spaces and properties to file
        :param game_elements:
        :param file_path:
        :return:
        """
        file = open(file_path, "a")
        # Dice#
        for i in game_elements['die_sequence']:
            if type(i) == list:
                file.write('Dice => ' + str(i) + '\n')

        # for i, his in enumerate(game_elements['history']['param']):
        #     if 'card' in his:
        #         if 'pack' in his:
        #             file.write('C' + str(his['pack'])[1:] +' Card => ' + his['card'].name + '\n')
        #             # print(i['pack'])
        #         else:
        #             if his['card'] in his['current_gameboard']['community_chest_cards']:
        #                 file.write('Community_chest' + ' Card => ' + his['card'].name + '\n')
        #             else:
        #                 file.write('Chance' + ' Card => ' + his['card'].name + '\n')
        #
        #         file.write(his['card'].name + ' is classified as ' + his['card'].card_type  + '\n')

                # if 'amount' in game_elements['history']['param'][i-1]:
                #     if 'go-to-nearest' not in his['card'].name:
                #         file.write(his['card'].name + ' is cost at ' + str(game_elements['history']['param'][i-1]['amount']) + '\n')

        # Add property info to file
        for loc in range(len(game_elements['location_sequence'])):
            space = game_elements['location_sequence'][int(loc)]
            # print(space, space.name,space.start_position)
            # for pos in range(space.start_position, space.end_position):
            space_dict = game_elements['locations'][space]
            file.write(self.mapping(space) + ' is located at ' + str(loc) + '\n')
            loc_class = space_dict['loc_class']
            file.write(self.mapping(space) + ' is classified as ' + str(loc_class) + '\n')

            if loc_class == 'real_estate':
                file.write(self.mapping(space) + ' is colored as ' + str(space_dict['color']) + '\n')
                file.write(self.mapping(space) + ' is price-1-house at ' + str(int(space_dict['price_per_house'])) + '\n')
                file.write(self.mapping(space) + ' is rented-1-hotel at ' + str(int(space_dict['rent_hotel'])) + '\n')
                file.write(self.mapping(space) + ' is rented-0-house at ' + str(int(space_dict['rent'])) + '\n')
                file.write(self.mapping(space) + ' is rented-1-house at ' + str(int(space_dict['rent_1_house'])) + '\n')
                file.write(self.mapping(space) + ' is rented-2-house at ' + str(int(space_dict['rent_2_houses'])) + '\n')
                file.write(self.mapping(space) + ' is rented-3-house at ' + str(int(space_dict['rent_3_houses'])) + '\n')
                file.write(self.mapping(space) + ' is rented-4-house at ' + str(int(space_dict['rent_4_houses'])) + '\n')

            if loc_class in ['real_estate', 'railroad',  'utility']:
                file.write(self.mapping(space) + ' is mortgaged at ' + str(int(space_dict['mortgage'])) + '\n')
                file.write(self.mapping(space) + ' is priced at ' + str(int(space_dict['price'])) + '\n')

            if loc_class == 'tax':
                file.write(self.mapping(space) + ' is cost at ' + str(int(space_dict['amount_due'])) + '\n')

        # Add go_increment to the file
        # file.write('GO is incremtent as ' + str(game_elements['go_increment']) + '\n')
            if space == 'Go':
                file.write('GO is located at ' + str(loc) + '\n')

        file.close()


    def set_board(self, gameboard):
        """
        The board may change after novelty injection.
        Hence, we need to update many parameters before each game
        :param gameboard:
        :return:
        """
        # Set the full board
        self.board_state = gameboard['location_sequence']
        # Set the property which can build property
        board_building = []
        for space in self.board_state:
            if gameboard['locations'][space]['loc_class'] == 'real_estate':
                board_building.append(space)
        self.board_building = board_building

        # Set the property can be owned
        board_owned = []
        for space in self.board_state:
            if gameboard['locations'][space]['loc_class'] in ['real_estate', 'railroad', 'utility']:
                board_owned.append(space)
        self.board_owned = board_owned

    #state_space = 28+22+n+n+2 = 56
    def board_to_state(self, current_board):
        """
        Represent the gameboard to a vector
        :param current_board: dict()
        :return: np.array
        """
        # print(current_board.keys())
        # print(current_board['locations'])
        state_space = []

        # Ownership of property => # of board states
        # -1 means other players, 0 means bank, and 1 means agent/ player 1
        state_space_owned = []
        self.board_state = [name.replace('-', ' ') for name in self.board_state]
        for space in self.board_state:
            if current_board['locations'][space]['loc_class'] != 'do_nothing' and \
                    current_board['locations'][space]['loc_class'] != 'action' and \
                    current_board['locations'][space]['loc_class'] != 'tax':
                if current_board['locations'][space]['owned_by'] == 'bank':
                    state_space_owned.append(0)
                elif current_board['locations'][space]['owned_by'] == 'player_1':
                    state_space_owned.append(1)
                else:
                    state_space_owned.append(-1)
            else:
                state_space_owned.append(0)
        state_space += state_space_owned

        # # Number of house in the property => # of houses in the space: 0,1,2,3,4,5
        # state_space_building = []
        # for space in self.board_state:
        #     if current_board['locations'][space]['loc_class'] == 'real_estate':
        #         if current_board['locations'][space]['num_hotels'] == 0:
        #             state_space_building.append(current_board['locations'][space]['num_houses'])
        #         else:
        #             state_space_building.append(5) # 5 denotes hotel
        #     else:
        #         state_space_building.append(0)
        # state_space += state_space_building

        # Position => n positions of players n = # of players
        # sorted_player = sorted(current_board['players'], key=lambda player: int(player.player_name[-1]))
        state_space_position = [0 for i in range(len(self.board_state))]
        if current_board['players']['player_1']['current_position']:
            state_space_position[current_board['players']['player_1']['current_position']] = 1
        state_space += state_space_position

        # # Card Ownership
        # #2 # of get-out_of_jail_card of players
        # state_space_card = []
        # com_card = [p.has_get_out_of_jail_community_chest_card for p in sorted_player]
        # chance_card = [p.has_get_out_of_jail_chance_card for p in sorted_player]
        # # first num denotes the # of card agent has
        # num_card_agent = int(com_card[0] + chance_card[0])
        # state_space_card.append(num_card_agent)
        # # second num denotes the cards other players have
        # num_card_others = sum(com_card) + sum(chance_card) - num_card_agent
        # state_space_card.append(num_card_others)
        # state_space += state_space_card

        # Cash
        # n cash ratio for all the players n = # of players
        sorted_player = ['player_1', 'player_2', 'player_3', 'player_4']
        state_space_cash = [0 for i in range(6 * len(current_board['players'].keys()))]
        for i, p in enumerate(sorted_player):
            if current_board['players'][p]['current_cash'] // 500 < 5:
                state_space_cash[int(i * 6 + current_board['players'][p]['current_cash']  // 500)] = 1
            else:
                state_space_cash[i * 6 + 5] = 1
        # state_space_cash = [int(p.current_cash) for p in sorted_player]
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
        """
        Action taken from A2C agent is a interger/index, need to transform to vector.
        :param action: a integer: index of action vector.
        :return: a vector with only one 1 otherwise 0 for the whole vector
        """
        actions_vec = [0 for i in range(self.action_space_num)]
        actions_vec[int(action)] = 1
        return actions_vec

    #action space 1+22+28+28+1 = 80
    def get_action_space(self, current_board, current_player):
        '''

        :param current_board: the gameboard with history.
        :param current_player: the player moving now: in our game, is player_1.
        :return:
        '''

        if self.site_space:
            if current_player['current_position'] != None:
                self.site_space[0] = current_player['current_position']
        else:

            # 1 first denotes the action -> buy or not buy
            self.action_space.append('buy_property')
            self.site_space.append(current_player['current_position'])

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
            self.action_space.append('concluded_actions')
            self.site_space.append(0)

        return self.action_space, self.site_space

        #take into array/vector and output the actions
    actions_vector_default = [1] + [0]

    def vector_to_actions(self, current_board, current_player, actions_vector=actions_vector_default):
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
                move_actions.append((self.action_space[i], self.board_state[self.site_space[i]]))

        self.move_actions = move_actions
        return self.move_actions
        #####becky - may be deleted##########
        # allowed_types = [location.UtilityLocation, location.RailroadLocation, location.RealEstateLocation]
        # if type(move_actions[0][1]) in allowed_types:
        #     return self.move_actions
        # else:
        #     return []

    def identify_free_mortgage(self,player):
        potentials = list()
        for a in player.assets:
            if a.is_mortgaged and a.mortgage <= player.current_cash:
                potentials.append(a)
        return potentials



