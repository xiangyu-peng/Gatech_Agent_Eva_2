import sys
sys.path.append('/media/becky/GNOME-p3/monopoly_simulator')
from gameplay_simple_becky_v1 import *
from gameplay_simple_tf import *
from interface import Interface
import simple_background_agent_becky_v1
from gym.utils import seeding
from random import randint
from location import *
from gym import error, spaces
import copy
# from agent_helper_functions import *
class Monopoly_world():
    def __init__(self):

        self.game_elements = None
        self.num_active_players = 2
        self.num_die_rolls = 0
        self.current_player_index = 0
        self.a = Interface()
        self.params = dict()
        self.player_decision_agents = dict()
        self.reward = 0
        self.terminal = False
        self.player_decision_agents = dict()
        self.seeds = 0
        self.done_indicator = 0
        self.win_indicator = 0
        self.die_roll = []

    def init(self):
        self.num_players = 2
        self.num_active_players = 2
        self.num_die_rolls = 0
        self.current_player_index = 0
        self.a = Interface()
        self.params = dict()
        self.player_decision_agents = dict()
        self.reward = 0
        self.terminal = False
        self.player_decision_agents = dict()
        self.game_elements = None
        self.seeds = self.seed(self.seeds + 1)


    def reset(self):
        self.init()
        player_list = ['player_' + str(i + 1) for i in range(self.num_active_players)]
        for player_name in player_list:
            self.player_decision_agents[player_name] = simple_background_agent_becky_v1.decision_agent_methods
        self.game_elements = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json', self.player_decision_agents, self.num_active_players)
        np.random.seed(self.seeds) #control the seed!!!!
        self.game_elements['seed'] = self.seeds
        self.game_elements['card_seed'] = self.seeds
        self.game_elements['choice_function'] = np.random.choice
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator = \
            before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)
        self.a.board_to_state(self.game_elements)
        return self.a.state_space, self.a.masked_actions

    def cal_asset_value(self, index):
        value = 0
        player = self.game_elements['players'][index]
        if player.assets:
            for asset in player.assets:

                #consider the num of houses in one space
                if type(asset) == RealEstateLocation:
                    num_houses = asset.num_houses
                    num_hotels = asset.num_hotels
                    value += asset.mortgage
                    if num_houses == 0 and num_hotels == 0:
                        value += asset.price + 1
                    elif num_houses == 1 and num_hotels == 0:
                        value += 5 * asset.rent_1_house
                    elif num_houses == 2 and num_hotels == 0:
                        value += 5 * asset.rent_2_house
                    elif num_houses == 3 and num_hotels == 0:
                        value += 5 * asset.rent_3_house
                    elif num_houses == 4 and num_hotels == 0:
                        value += 5 * asset.rent_4_house
                    elif num_hotels == 1:
                        value += 5 * asset.rent_hotel
                else:
                    value += asset.price
        return value



    # def reward_cal(self, win_indicator):
    #     reward = self.game_elements['players'][self.current_player_index].current_cash + self.cal_asset_value(self.current_player_index)
    #     rewards_total = 0
    #     for num in range(self.num_players):
    #         rewards_total += self.game_elements['players'][num].current_cash
    #         rewards_total += self.cal_asset_value(num)
    #     return reward / (rewards_total + 0.1) + win_indicator

        # return 1 + win_indicator * 100
    def reward_cal(self, win_indicator, action_num):
        if action_num == 0: #buy
            return 5 + 15 * win_indicator
        elif action_num == 79: #skip
            return 1 + 100 * win_indicator
        else:
            return 2 + 15 * win_indicator



    def next(self, action):
        action_num = action
        action = self.a.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
            after_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, self.a, self.params)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                simulate_game_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator = \
                before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)

        self.terminal = 0 if self.num_active_players > 1 else 1
        if self.done_indicator == 1:
            self.terminal = 1
        self.reward = self.reward_cal(self.win_indicator, action_num)
        if self.terminal:
            if self.win_indicator == 1:
                self.terminal = 2
        state_space = self.a.board_to_state(self.game_elements)
        reward = self.reward
        terminal = self.terminal
        masked_actions = self.a.masked_actions

        return state_space, reward, terminal, masked_actions #can put KG in info


    def next_after_nochange(self, action):
        action_num = action
        action = self.a.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
            after_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, self.a, self.params)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                simulate_game_step_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.die_roll)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator = \
                before_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.die_roll)

        self.terminal = 0 if self.num_active_players > 1 else 1
        if self.done_indicator == 1:
            self.terminal = 1
        self.reward = self.reward_cal(self.win_indicator, action_num)
        if self.terminal:
            if self.win_indicator == 1:
                self.terminal = 2

        state_space = self.a.board_to_state(self.game_elements)
        reward = self.reward
        terminal = self.terminal
        masked_actions = self.a.masked_actions

        return state_space, reward, terminal, masked_actions #can put KG in info

    def next_nochange(self, action):

        #not change#
        game_elements_ori = copy.deepcopy(self.game_elements)
        num_active_players_ori = self.num_active_players
        # num_die_rolls_ori = self.num_die_rolls
        current_player_index_ori = self.current_player_index
        done_indicator_ori = self.done_indicator
        win_indicator_ori = self.win_indicator
        ######

        action_num = action
        a = Interface()
        action = a.action_num2vec(action)
        game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
            after_agent_tf_nochange(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, a, self.params)
        if num_active_players > 1:
            game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, self.die_roll = \
                simulate_game_step_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index)
        if num_active_players > 1:
            game_elements, num_active_players, num_die_rolls, current_player_index, a, params, self.die_roll = \
                before_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, a, self.die_roll)

        terminal = 0 if num_active_players > 1 else 1
        if done_indicator == 1:
            terminal = 1
        reward = self.reward_cal(win_indicator, action_num)
        if terminal:
            if win_indicator == 1:
                terminal = 2
        state_space = self.a.board_to_state(self.game_elements)
        masked_actions = a.masked_actions


        self.game_elements = game_elements_ori
        self.num_active_players = num_active_players_ori
        # self.num_die_rolls = num_die_rolls_ori
        self.current_player_index = current_player_index_ori
        self.done_indicator = done_indicator_ori
        self.win_indicator = win_indicator_ori

        return state_space, reward, terminal, masked_actions #can put KG in info

    def seed(self, seed=None):
        np_random, seed1 = seeding.np_random(seed)
        self.seeds = seeding.hash_seed(seed1 + 1) % 2 ** 31
        # self.seeds = seed
        return self.seeds