import sys
import os
curr_path = os.getcwd()
curr_path = curr_path.replace("/monopoly_simulator", "")
sys.path.append(curr_path + '/env')
sys.path.append(curr_path + '/KG-rule')
from gameplay_step import *
from gameplay_tf import *
from interface import Interface
from simple_background_agent_becky_p1 import P1Agent
from simple_background_agent_becky_p2 import P2Agent
from background_agent_v2 import P2Agent_v2
from gym.utils import seeding
from random import randint
from location import *
from gym import error, spaces
import copy
import logging
from log_setting import set_log_level, ini_log_level
from openie_triple import KG_OpenIE
from configparser import ConfigParser

# from agent_helper_functions import *
class Monopoly_world():
    def __init__(self):
        #get config data
        config_file = '/media/becky/GNOME-p3/monopoly_simulator/config.ini'
        config_data = ConfigParser()
        config_data.read(config_file)
        self.hyperparams = self.params_read(config_data)
        self.game_elements = None
        self.num_active_players = self.hyperparams['num_active_players']
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
        self.masked_actions = []
        self.kg = KG_OpenIE()
        self.kg_save_num = 0
        self.kg_save_interval = self.hyperparams['kg_save_interval']
        self.log_path = self.hyperparams['log_path']
        self.env_num = 0
        self.value_past = self.hyperparams['initial_cash']
        self.value_total = 2 * self.hyperparams['initial_cash']
        self.kg_change = []


    def params_read(self, config_data):
        params = {}
        # print('config_data',config_data.options('env'))
        for key in config_data['env']:
            v = eval(config_data['env'][key])
            params[key] = v
        # print('params',params)
        return params

    def init(self):
        ini_log_level()
        set_log_level()
        self.num_players = self.hyperparams['num_active_players']
        self.num_active_players = self.hyperparams['num_active_players']
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
        self.env_num = self.env_num + 1
        self.value_past = self.hyperparams['initial_cash']
        self.value_total = 2 * self.hyperparams['initial_cash']


    def reset(self):
        self.init()
        # if os.path.exists(self.log_path):
        #     os.remove(self.log_path)
        logger = set_log_level()
        # logger.info('seed is ' + str(self.seeds))
        self.player_decision_agents['player_1'] = P1Agent()
        self.player_decision_agents['player_2'] = P2Agent_v2()

        self.game_elements = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json', self.player_decision_agents, self.num_active_players)

        if self.env_num > 1000:
            inject_novelty(self.game_elements)

        np.random.seed(self.seeds) #control the seed!!!!
        self.game_elements['seed'] = self.seeds
        self.game_elements['card_seed'] = self.seeds
        self.game_elements['choice_function'] = np.random.choice
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator, masked_actions = \
            before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)
        self.a.board_to_state(self.game_elements)
        self.masked_actions = masked_actions

        return self.a.state_space, self.masked_actions

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
                        value += asset.price * 1.1
                    elif num_houses == 1 and num_hotels == 0:
                        value += 5 * asset.rent_1_house + asset.price + 100
                    elif num_houses == 2 and num_hotels == 0:
                        value += 5 * asset.rent_2_houses + asset.price + 100
                    elif num_houses == 3 and num_hotels == 0:
                        value += 5 * asset.rent_3_houses + asset.price + 100
                    elif num_houses == 4 and num_hotels == 0:
                        value += 5 * asset.rent_4_houses + asset.price + 100
                    elif num_hotels == 1:
                        value += 5 * asset.rent_hotel + asset.price + 100
                else:
                    value += asset.price * 2
        return value



    # def reward_cal(self, win_indicator):
    #     reward = self.game_elements['players'][self.current_player_index].current_cash + self.cal_asset_value(self.current_player_index)
    #     rewards_total = 0
    #     for num in range(self.num_players):
    #         rewards_total += self.game_elements['players'][num].current_cash
    #         rewards_total += self.cal_asset_value(num)
    #     return reward / (rewards_total + 0.1) + win_indicator

        # return 1 + win_indicator * 100
    def reward_cal(self, win_indicator, action_num, masked_actions_reward):
        if masked_actions_reward[action_num] == 0:
            return -0.01
        else:
            value_now = self.game_elements['players'][self.current_player_index].current_cash +\
                     self.cal_asset_value(self.current_player_index)
            rewards_total = 0
            for num in range(self.num_players):
                rewards_total += self.game_elements['players'][num].current_cash
                rewards_total += self.cal_asset_value(num)
            reward = (value_now - self.value_past) / self.value_total
            self.value_total = rewards_total
            self.value_past = value_now
            if masked_actions_reward[(action_num + 1) % 2] == 0:
                return 0.01
            return reward



            # if action_num == 0: #buy
            #     return (reward * 1.2) / (rewards_total + 0.1) + win_indicator
            # else:
            #     return reward / (rewards_total + 0.1) + win_indicator
            # if action_num == 0: #buy
            #     return 1 + win_indicator
            # else:
            #     return 0 + win_indicator

        # if action_num == 0: #buy
        #     return 100 + 1000 * win_indicator
        # elif action_num == 79: #skip
        #     return 1 + 1000 * win_indicator
        # else:
        #     return 0 + 100 * win_indicator



    def next(self, action):
        '''
        This function is same with the one in gym env. Given an action and return the state, reward, done and info
        :param action: The action given by actor! But we won't consider if this action is valid or not.
        :return: state, reward, done and info/ masked_actions
        '''

        masked_actions = [1, 1] #we don't consider if action is valid or not in this function

        #When last state has a winner, we will reset the game
        if self.terminal == 1:
            self.save_kg()
            self.reset()
            state_space = self.a.board_to_state(self.game_elements)
            reward = 0
            terminal = 0
            masked_actions = self.a.masked_actions
        else:
        #     action_num = action
        #     masked_actions_reward = self.a.masked_actions.copy()
        #
        #     #When the action is not valid, the state won't change and the reward will be negative
        #     if masked_actions_reward[action_num] == 0:
        #         state_space = self.a.board_to_state(self.game_elements)
        #         self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        #         reward = self.reward
        #         terminal = self.terminal
        #         masked_actions = self.masked_actions
        #
        #     #When the action from agent is valid, the state will go to next step
        #     else:
        #         action = self.a.action_num2vec(action)
        #         self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
        #             after_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, self.a, self.params)
        #         if self.num_active_players > 1:
        #             self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
        #                 simulate_game_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index)
        #         if self.num_active_players > 1:
        #             self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator, masked_actions = \
        #                 before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)
        #
        #         self.terminal = 0 if self.num_active_players > 1 else 1
        #         if self.done_indicator == 1:
        #             self.terminal = 1
        #         self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        #         if self.terminal:
        #             if self.win_indicator == 1:
        #                 self.terminal = 2
        #         state_space = self.a.board_to_state(self.game_elements)
        #         reward = self.reward
        #         terminal = self.terminal
        #         self.masked_actions = masked_actions
        #
        # return state_space, reward, terminal, masked_actions #can put KG in info


            action_num = action
            masked_actions_reward = self.a.masked_actions.copy()
            action = self.a.action_num2vec(action)
            self.die_roll = []

            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, params, self.masked_actions, done_hyp = \
                after_agent_hyp(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index,
                            action, self.a, self.params)
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                after_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params)
            if self.num_active_players > 1:
                self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                    simulate_game_step_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.die_roll)
            if self.num_active_players > 1:
                self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator, masked_actions = \
                    before_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.die_roll)

            self.terminal = 0 if self.num_active_players > 1 else 1
            if self.done_indicator == 1:
                self.terminal = 1
            self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
            if self.terminal:
                if self.win_indicator == 1:
                    self.terminal = 2

            state_space = self.a.board_to_state(self.game_elements)
            reward = self.reward
            terminal = self.terminal

        return state_space, reward, terminal, masked_actions #can put KG in info


    def next_after_nochange(self, action):
        masked_actions = [1, 1]
        # if self.terminal == 1:
        #     self.reset()
        #     state_space = self.a.board_to_state(self.game_elements)
        #     reward = 0
        #     terminal = 0
        #     masked_actions = self.a.masked_actions
        # else:
        action_num = action
        masked_actions_reward = self.a.masked_actions.copy()
        # When the action is not valid, the state won't change and the reward will be negative
        # if masked_actions_reward[action_num] == 0:
            # state_space = self.a.board_to_state(self.game_elements)
            # self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
            # reward = self.reward
            # terminal = self.terminal
            # masked_actions = self.masked_actions

        # In tf part, even though the action is invalid, we will still go to next step, because we cannot
        # be obstructed in one state
        action = self.a.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
            after_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                simulate_game_step_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.die_roll)
        if self.num_active_players > 1:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params, self.win_indicator, masked_actions = \
                before_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.die_roll)

        self.terminal = 0 if self.num_active_players > 1 else 1
        if self.done_indicator == 1:
            self.terminal = 1
        self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        if self.terminal:
            if self.win_indicator == 1:
                self.terminal = 2

        state_space = self.a.board_to_state(self.game_elements)
        reward = self.reward
        terminal = self.terminal

        if self.terminal > 0:
            self.save_kg()
            self.reset()
            state_space = self.a.board_to_state(self.game_elements)
            reward = 0
            terminal = 0
            masked_actions = self.a.masked_actions

            if self.env_num >= 20:
                info = (masked_actions, self.kg.kg_vector)
            else:
                info = (masked_actions,[])
        else:
            info = (masked_actions,[])
        return state_space, 0, terminal, info  #can put KG in info

    def next_hyp(self, action):
        # masked_actions = [1, 1]
        action_num = action
        masked_actions_reward = self.a.masked_actions.copy()
        # # When the action is not valid, the state won't change and the reward will be negative
        # if masked_actions_reward[action_num] == 0:
        #     state_space = self.a.board_to_state(self.game_elements)
        #     reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        #     terminal = self.terminal
        #     masked_actions = self.masked_actions
        #     self.die_roll = []

        a = Interface()
        action = a.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, params, self.masked_actions, done_hyp = \
            after_agent_hyp(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, a,
                            self.params)


        reward = self.reward_cal(0, action_num, masked_actions_reward)
        state_space = self.a.board_to_state(self.game_elements)

        return state_space, reward, done_hyp, self.masked_actions  # can put KG in in

    def next_nochange(self, action):
        # if self.terminal == 1:
        #     self.reset()

        #not change#
        game_elements_ori = copy.deepcopy(self.game_elements)
        num_active_players_ori = self.num_active_players
        # num_die_rolls_ori = self.num_die_rolls
        current_player_index_ori = self.current_player_index
        done_indicator_ori = self.done_indicator
        win_indicator_ori = self.win_indicator
        ######

        masked_actions = [1, 1]
        action_num = action
        masked_actions_reward = self.a.masked_actions.copy()
        # When the action is not valid, the state won't change and the reward will be negative
        if masked_actions_reward[action_num] == 0:
            state_space = self.a.board_to_state(self.game_elements)
            reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
            terminal = self.terminal
            masked_actions = self.masked_actions
            self.die_roll = []

        else:
            a = Interface()
            action = a.action_num2vec(action)
            game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
                after_agent_tf_nochange(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, a, self.params)
            if num_active_players > 1:
                game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, self.die_roll = \
                    simulate_game_step_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index)
            if num_active_players > 1:
                game_elements, num_active_players, num_die_rolls, current_player_index, a, params, self.die_roll, win_indicator, masked_actions = \
                    before_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, a, self.die_roll)

            terminal = 0 if num_active_players > 1 else 1
            if done_indicator == 1:
                terminal = 1
            reward = self.reward_cal(win_indicator, action_num, masked_actions_reward)
            if terminal:
                if win_indicator == 1:
                    terminal = 2
            state_space = self.a.board_to_state(self.game_elements)


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

    def save_kg(self):
        self.kg_change = self.kg.build_kg_file(self.log_path, level='rel', use_hash=True)
        # file = open(self.log_path,'r')
        # for line in file:
        #     kg_change = self.kg.build_kg(line, level='rel', use_hash=True)

        #TODO: change name to config
        # if self.kg_change:
        #     file = open("/media/becky/GNOME-p3/KG-rule/test.txt", "a")
        #     file.write(str(self.kg_change) + ' \n')
        #     file.close()

        self.kg_save_num += 1
        #save knowledge graph when simulating num of games is self.kg_save_interval
        if self.kg_save_num % self.kg_save_interval == 0:
            self.kg.save_json(level='rel')
            self.kg.dict_to_matrix()
            self.kg.save_matrix()
            self.kg.save_vector()
            # file = open("/media/becky/GNOME-p3/KG-rule/test.txt", "a")
            # if self.kg_change == []:
            #     file.write('None' + ' \n')

            file.write(str(self.kg_change) + ' \n')
            file.close()




