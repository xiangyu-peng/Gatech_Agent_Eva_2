import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator','')
upper_path_eva = upper_path + '/Evaluation/monopoly_simulator'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
sys.path.append(upper_path_eva)
####################

from monopoly_simulator_background.gameplay_step import *
from monopoly_simulator_background.gameplay_tf import *
from monopoly_simulator_background.interface import Interface
from GNN.KG_state import KG_Interface
from monopoly_simulator_background.simple_background_agent_becky_p1 import P1Agent
from monopoly_simulator_background.simple_background_agent_becky_p2 import P2Agent
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent
from monopoly_simulator import action_choices
from gym.utils import seeding
from random import randint
from monopoly_simulator.location import *
from gym import error, spaces
import copy
import logging
from monopoly_simulator_background.log_setting import set_log_level, ini_log_level
from KG_rule.openie_triple import KG_OpenIE
from configparser import ConfigParser
from monopoly_simulator import player
from monopoly_simulator import read_write_current_state
from monopoly_simulator import initialize_game_elements
from monopoly_simulator.read_write_current_state import write_out_current_state_to_file, read_in_current_state_from_file
from monopoly_simulator.hypothetical_simulator import initialize_hypothetical_universe
# from monopoly_simulator_background.agent_helper_functions import *
from monopoly_simulator_background.agent_helper_functions import identify_improvement_opportunity_all


class Monopoly_world():
    def __init__(self, config_file=None):
        # Get config data
        # exp related
        self.novelty_num = None
        self.matrix_name = None
        self.entity_name = None
        self.novelty_inject_num = 0

        # self.upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator','')
        self.upper_path = '/media/becky/GNOME-p3'
        self.config_file = self.upper_path + config_file
        config_data = ConfigParser()
        config_data.read(self.config_file)
        self.hyperparams = self.params_read(config_data, 'env')

        # env parameters
        self.game_elements = None
        self.num_players = 0
        self.num_active_players = 0
        self.num_die_rolls = 0
        self.current_player_index = 0
        self.interface = None
        self.exp_type = None
        self.params = dict()
        self.player_decision_agents = dict()
        self.reward = 0
        self.terminal = False  # the game is end or not
        self.seeds = 0
        self.done_indicator = 0  # the indicator indicating done or not
        self.win_indicator = 0  # win or lose
        self.die_roll = []  # TF use, record the dice roll num
        self.masked_actions = []
        # self.value_past = self.hyperparams['initial_cash']
        # self.value_total = 2 * self.hyperparams['initial_cash']
        self.avg_die = 0  # dice average roll sum
        self.action_background = None
        self.reward_diff = 0

        # Rule learning
        self.kg_use = True  # True: learn kg, False: no kg rule
        self.gameboard_initial = None  # A gameboard set to this env, but we assign our agent to players
        self.gameboard_set_ini = None  # A gameboard set to this env
        self.kg = None  # kg class
        self.kg_save_num = 0
        self.game_num = 0
        self.kg_change = []
        self.kg_save_interval = self.hyperparams['kg_save_interval']
        self.log_path = None
        self.rule_change_path = self.upper_path + self.hyperparams['rule_change_path']
        self.kg_rel_path = self.upper_path + self.hyperparams['kg_rel_path']

        #hypothetical simulator
        self.saved_gameboard_path = None
        self.running_hyp = False

    def set_exp(self, exp_dict):
        if exp_dict:
            self.novelty_num = exp_dict['novelty_num']  # (5,1) (a,b): 1,2,3,4,5 => range(a, a+b)
            self.matrix_name = '/kg_matrix_' + str(exp_dict['novelty_num'][0]) + '_' + str(exp_dict['novelty_num'][1]) + '_' + exp_dict['exp_type'] + '.npy'
            self.entity_name = '/entity_id_' + str(exp_dict['novelty_num'][0]) + '_' + str(exp_dict['novelty_num'][1]) + '_' + exp_dict['exp_type'] + '.json'
            self.novelty_inject_num = exp_dict['novelty_inject_num']
            self.log_path = self.upper_path + '/KG_rule/log_file/game_log_' + str(exp_dict['novelty_num'][0]) + '_' + str(exp_dict['novelty_num'][1]) + '_' + exp_dict['exp_type'] + '.txt'
            self.exp_type = exp_dict['exp_type']
            if exp_dict['exp_type'] == 'state':
                self.interface = KG_Interface()
            else:
                self.interface = Interface()
        else:
            self.interface = Interface()

    def set_initial_gameboard(self, gameboard=None):
        # If we run the hypothetical simulator, the gameboard is written to a file
        if isinstance(gameboard, str):  # this trigger the hypothetical simulation, gameboard is a file path
            self.num_players = self.hyperparams['num_active_players']
            self.player_decision_agents['player_1'] = P1Agent()
            name_num = 1
            while name_num < self.num_players:
                name_num += 1
                self.player_decision_agents['player_' + str(name_num)] = P2Agent() #Agent(**background_agent_v3.decision_agent_methods)
            self.gameboard_initial = read_in_current_state_from_file(self.upper_path+gameboard, self.player_decision_agents)
            # TODO: seed
            self.running_hyp = True

        else:
            # If we assign a gameboard instead of using the default one
            if gameboard:
                self.gameboard_initial = copy.deepcopy(gameboard)
                self.num_players = len(self.gameboard_initial['players'])  # Update # of players

                # Reset the player.agents ##########################
                # Set up agents back to the agents we can use
                self.player_decision_agents['player_1'] = P1Agent()

                # Assign the beckground agents to the players
                name_num = 1
                while name_num < self.num_players:
                    name_num += 1
                    self.player_decision_agents['player_' + str(name_num)] = P2Agent() #Agent(**background_agent_v3.decision_agent_methods)
                #####################################################

                # set player.agent to  the gameboard
                self.gameboard_initial['players'] = dict()
                game_board_schema = json.load(open(self.upper_path + '/monopoly_game_schema_v1-1.json', 'r'))
                game_board_schema['players']['player_states']['player_name'] = \
                    game_board_schema['players']['player_states']['player_name'][: self.num_players]  # change the player_num
                initialize_game_elements._initialize_players(
                    self.gameboard_initial,
                    game_board_schema,
                    self.player_decision_agents)  # json path here doesn't matter

            # OR We use the default gameboard
            else:
                self.num_players = self.hyperparams['num_active_players']
                # Reset the player.agents ##########################
                # Set up agents
                self.player_decision_agents['player_1'] = P1Agent()

                # Assign the beckground agents to the players
                name_num = 1
                while name_num < self.num_players:
                    name_num += 1
                    self.player_decision_agents['player_' + str(name_num)] = P2Agent() #Agent(**background_agent_v3.decision_agent_methods)

                self.gameboard_initial = set_up_board(self.upper_path + '/monopoly_game_schema_v1-1.json',
                                                      self.player_decision_agents,
                                                      self.num_players)
                #####################################################

            # In case of the gameboard we use to assign here has history, we have to clear the history here
            if 'seed' in self.gameboard_initial:
                self.gameboard_initial.pop('seed')
            if 'card_seed' in self.gameboard_initial:
                self.gameboard_initial.pop('card_seed')
            if 'choice_function' in self.gameboard_initial:
                self.gameboard_initial.pop('choice_function')

        # KG for rule learning
        if self.kg_use:
            self.kg = KG_OpenIE(self.gameboard_initial, self.matrix_name, self.entity_name, config_file=self.config_file)

        return self.gameboard_initial

    def params_read(self, config_data, key_parameters):
        """
        Read the config.ini file
        """
        params = {}
        for key in config_data[key_parameters]:
            v = eval(config_data[key_parameters][key])
            params[key] = v
        return params

    def init(self):
        """
        RESET all the parameters before each game
        """
        self.num_active_players = self.num_players  # Reset the active players number to players number
        self.num_die_rolls = 0
        self.current_player_index = 0
        self.interface = KG_Interface() if self.exp_type == 'state' else Interface()
        self.params = dict()
        self.reward = 0
        self.terminal = False
        self.seeds = self.seed(self.seeds + 1)  # Change to another seed! So we won't run the same game again
        self.game_num = self.game_num + 1  # Record # of games we run
        # self.value_past = self.hyperparams['initial_cash']
        # self.value_total = 2 * self.hyperparams['initial_cash']
        self.game_elements = copy.deepcopy(self.gameboard_initial)
        self.reward_diff = 0


    def reset(self):
        """
        RESET the game
        """
        self.init()  # Reset all the parameters
        self.interface.set_board(self.gameboard_initial)

        if self.kg_use:
            self.kg.set_gameboard(self.game_elements)  # Deliver gameboard info to openie

        # Inject novelty here
        if self.game_num > self.novelty_inject_num:
            inject_novelty(self.game_elements, self.novelty_num)

        if self.running_hyp:
            current_player = self.game_elements['players'][0]
            state = self.interface.board_to_state(self.game_elements)
            allowable_actions = current_player.compute_allowable_post_roll_actions(self.game_elements)
            params_mask = identify_improvement_opportunity_all(current_player, self.game_elements)
            self.masked_actions = self.interface.get_masked_actions(allowable_actions, params_mask, self.game_elements['players'][0])
            np.random.seed(self.seeds)  # control the seed!!!!
            self.game_elements['seed'] = self.seeds
            self.game_elements['card_seed'] = self.seeds
            self.game_elements['choice_function'] = np.random.choice
            self.num_active_players = self.num_active_players - sum([1 if player.status == 'lost'\
                else 0 for player in self.game_elements['players']])  # some players may lose already

        else:
            np.random.seed(self.seeds)  # control the seed!!!!
            self.game_elements['seed'] = self.seeds
            self.game_elements['card_seed'] = self.seeds
            self.game_elements['choice_function'] = np.random.choice
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.interface, self.params, self.win_indicator, masked_actions = \
                before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.interface)
            state = self.interface.board_to_state(self.game_elements)
            self.masked_actions = masked_actions

        # calculate average sum of dice for calculating rewards
        self.avg_die = sum([sum(i.die_state)/len(i.die_state) for i in self.game_elements['dies']])

        self.action_background = self.background_agent_decision(self.masked_actions)
        self.reward_diff = self.reward_cal(win_indicator=0, action_num=None, masked_actions_reward=None)

        return state, (self.masked_actions, self.action_background)


    def get_income(self, index):
        """
        Calculate the income of every player
        :param index: The index of the player, i.e. 0
        :return: int: income of one player
        """
        value = 0
        player = self.game_elements['players'][index]
        if player.assets:
            for asset in player.assets:
                if asset.is_mortgaged:
                    pass
                else:
                    #consider the num of houses in one space
                    if type(asset) == RealEstateLocation:
                        value += asset.calculate_rent()
                    elif type(asset) == RealEstateLocation:
                        value += asset.calculate_railroad_dues()
                    elif type(asset) == UtilityLocation:
                        value += asset.calculate_utility_dues(self.avg_die)
        return value


    def get_money_diff(self, player_index, opponents_index):
        """
        Normalize the difference of players' cash
        :param player_index: player index of our agent
        :param opponents_index: index of other players
        :return: float: normalized diff
        """
        player = self.game_elements['players'][player_index]
        opponents = [self.game_elements['players'][i] for i in opponents_index]
        player_money_norm = self.normalize(player.current_cash, x_min=0, x_max=10000, a=0)
        opps_money_norm = self.normalize(opponents[0].current_cash, x_min=0, x_max=10000, a=0) if len(opponents) == 1 else 0
        return (player_money_norm - opps_money_norm) / max(player_money_norm, opps_money_norm)


    def get_income_diff(self, player_index, opponents_index):
        """
        Normalize the difference of players' income
        :param player_index: player index of our agent
        :param opponents_index: index of other players
        :return: float: normalized diff
        """
        # player = self.game_elements['players'][player_index]
        # opponents = [self.game_elements['players'][i] for i in opponents_index]
        player_income_norm = self.normalize(self.get_income(player_index), x_min=0, x_max=2000, a=0)
        opps_income_norm = self.normalize(self.get_income(1), x_min=0, x_max=2000, a=0) if len(
            opponents_index) == 1 else 0
        return (player_income_norm - opps_income_norm) / max(player_income_norm, opps_income_norm, 0.000000000001)


    def normalize(self, x, x_min, x_max, a=-1, b=1):
        """
        :param x: value
        :param x_min: init range min
        :param x_max: init range max
        :param a: result range min
        :param b: result range max
        :return: normalized value
        """
        value = (x - x_min) / (x_max - x_min) * (b - a) + a
        value = np.clip(value, a, b)
        return value


    def reward_cal(self, win_indicator=0, action_num=None, masked_actions_reward=None):
        """
        Calculating rewards of each state
        :param win_indicator: win or lose
        :param action_num: action took
        :param masked_actions_reward: masked actions vector for calculating rewards i.e. [0,1]
        :return: float : reward
        """
        # If this action is invalid:
        # value_now = self.game_elements['players'][self.current_player_index].current_cash +\
        #          self.cal_asset_value(self.current_player_index)
        # rewards_total = 0
        # for num in range(self.num_players):
        #     rewards_total += self.game_elements['players'][num].current_cash
        #     rewards_total += self.cal_asset_value(num)
        # reward = (value_now - self.value_past) / self.value_total

        # Calculating rewards
        money_diff = self.get_money_diff(0, [1])
        income_diff = self.get_income_diff(0, [1])
        reward = money_diff * 0.5 + income_diff * 0.5

        if masked_actions_reward == None:
            return reward

        if masked_actions_reward[action_num] == 0:
            self.reward_diff = reward
            return -0.01

        reward, self.reward_diff = reward - self.reward_diff, reward
        # We give a little more rewards to but_property action
        # if action_num == 0:
        #     reward += 0.001

        # self.value_total = rewards_total
        # self.value_past = value_now

        # if the action is the same with the one of background agent, we will have a bonus
        if self.action_background == action_num and action_num == 0:
            reward += 0.1
        elif self.action_background == action_num and action_num == 1:
            reward += 0.01
        # If this action is the only action we can choose, we give it a little bonus
        if masked_actions_reward[(action_num + 1) % 2] == 0:
            return max(0.001, reward)

        return reward

    # Hypothetical training
    # def next_hypothetical_training(self, action):
    #     # Read the gameboard first
    #     current_board = read_in_current_state_from_file(infile, player_decision_agents)
    #     print('lll')

    def next(self, action):
        '''
        This function is same with the one in gym env. Given an action and return the state, reward, done and info
        :param action: The action given by actor! But we won't consider if this action is valid or not.
        :return: state, reward, done and info/ masked_actions
        '''

        # masked_actions = [1, 1] #we don't consider if action is valid or not in this function

        #When last state has a winner, we will reset the game
        if self.terminal > 0:
            if self.kg_use:
                self.interface.get_logging_info(self.game_elements, current_player_index=0,file_path=self.log_path)
                self.save_kg()
                self.interface.clear_history(file_path=self.log_path)
            state_space, (masked_actions, action_background) = self.reset()
            reward = 0
            terminal = 0

        else:
            action_num = action
            masked_actions_reward = self.interface.masked_actions.copy()
            action = self.interface.action_num2vec(action)
            self.die_roll = []

            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                after_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls,
                                    self.current_player_index, action, self.interface, self.params)

            if self.win_indicator == 0:
                loop_num = 1
                while loop_num < self.num_players:
                    loop_num += 1
                    self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                        simulate_game_step_tf_step(self.game_elements, self.num_active_players,
                                                   self.num_die_rolls, self.current_player_index,
                                                   self.die_roll, self.done_indicator, self.win_indicator,
                                                   self.interface)

            if self.win_indicator == 0:
                self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.interface, self.params, self.win_indicator, masked_actions = \
                    before_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls,
                                         self.current_player_index, self.interface, self.die_roll)
            # save players  moving history
            if self.kg_use:
                self.interface.save_history(self.game_elements)

            self.terminal = 0 if self.win_indicator == 0 else 1

            if self.done_indicator == 1:
                self.terminal = 1
            self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)

            if self.terminal:
                if self.win_indicator == 1:
                    self.terminal = 2
                if self.win_indicator == 0:
                    self.terminal = 3

            state_space = self.interface.board_to_state(self.game_elements)
            reward = self.reward
            terminal = self.terminal
            masked_actions = self.interface.masked_actions
            if self.terminal:
                action_background = 1
            else:
                action_background = self.background_agent_decision(masked_actions)


        return state_space, reward, terminal, (masked_actions,action_background) #can put KG in info


    def next_after_nochange(self, action):
        masked_actions = [1, 1]
        action_num = action
        masked_actions_reward = self.interface.masked_actions.copy()
        # When the action is not valid, the state won't change and the reward will be negative
        # if masked_actions_reward[action_num] == 0:
            # state_space = self.interface.board_to_state(self.game_elements)
            # self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
            # reward = self.reward
            # terminal = self.terminal
            # masked_actions = self.masked_actions

        # In tf part, even though the action is invalid, we will still go to next step, because we cannot
        # be obstructed in one state
        action = self.interface.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
            after_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls,
                                self.current_player_index, action, self.interface, self.params)

        if self.win_indicator == 0:
            loop_num = 1
            while loop_num < self.num_players:
                loop_num += 1
                self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.done_indicator, self.win_indicator = \
                    simulate_game_step_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls,
                                               self.current_player_index, self.die_roll, self.done_indicator,
                                               self.win_indicator, self.interface)

        if self.win_indicator == 0:
            self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.interface, self.params, self.win_indicator, masked_actions = \
                before_agent_tf_step(self.game_elements, self.num_active_players, self.num_die_rolls,
                                     self.current_player_index, self.interface, self.die_roll)
        #save players moving history
        if self.kg_use:
            self.interface.save_history(self.game_elements)

        self.terminal = 0 if self.win_indicator == 0 else 1
        if self.done_indicator == 1:
            self.terminal = 1
        self.reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        if self.terminal:
            if self.win_indicator == 1:
                self.terminal = 2

        state_space = self.interface.board_to_state(self.game_elements)
        reward = self.reward
        terminal = self.terminal

        if self.terminal > 0:
            if self.kg_use:
                self.interface.get_logging_info(self.game_elements, current_player_index=0, file_path=self.log_path)
                self.save_kg()
                self.interface.clear_history(file_path=self.log_path)
            state_space, (masked_actions, action_background) = self.reset()
            # state_space = self.interface.board_to_state(self.game_elements)
            reward = 0
            terminal = 1
            # masked_actions = self.masked_actions

            info = (masked_actions, action_background)
            # if self.game_num >= 10:
            #     if self.kg_use:
            #         info = (masked_actions, self.kg.kg_vector)
        else:
            action_background = self.background_agent_decision(masked_actions)
            info = (masked_actions,action_background)
        self.action_background = action_background
        self.masked_actions = masked_actions
        return state_space, 0, terminal, info  #can put KG in info

    def next_hyp(self, action):
        # masked_actions = [1, 1]
        action_num = action
        masked_actions_reward = self.interface.masked_actions.copy()
        # # When the action is not valid, the state won't change and the reward will be negative
        # if masked_actions_reward[action_num] == 0:
        #     state_space = self.interface.board_to_state(self.game_elements)
        #     reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        #     terminal = self.terminal
        #     masked_actions = self.masked_actions
        #     self.die_roll = []

        self.interface.set_board(self.game_elements)
        action = self.interface.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, index, self.interface, params, self.masked_actions, done_hyp, done_indicator = \
            after_agent_hyp(self.game_elements, self.num_active_players, self.num_die_rolls,
                            self.current_player_index, action, self.interface, self.params)


        reward = self.reward_cal(0, action_num, masked_actions_reward)
        state_space = self.interface.board_to_state(self.game_elements)

        return state_space, reward, (done_hyp, done_indicator), self.masked_actions  # can put KG in in

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
        masked_actions_reward = self.masked_actions

        # When the action is not valid, the state won't change and the reward will be negative
        # if masked_actions_reward[action_num] == 0:
        #     state_space = self.interface.board_to_state(self.game_elements)
        #     reward = self.reward_cal(self.win_indicator, action_num, masked_actions_reward)
        #     terminal = self.terminal
        #     masked_actions = self.masked_actions
        #     self.die_roll = []

        # else:
        # a = Interface()
        # a.set_board(self.game_elements_)
        action = self.interface.action_num2vec(action)
        game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
            after_agent_tf_nochange(self.game_elements, self.num_active_players, self.num_die_rolls,
                                    self.current_player_index, action, self.interface, self.params)

        self.die_roll = []
        if win_indicator == 0:
            loop_num = 1
            while loop_num < self.num_players:
                loop_num += 1
                game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, self.die_roll = \
                    simulate_game_step_tf_nochange(game_elements, num_active_players, num_die_rolls,
                                                   current_player_index, done_indicator, win_indicator,
                                                   self.die_roll, self.interface)
        if win_indicator == 0:
            game_elements, num_active_players, num_die_rolls, current_player_index, a, params, self.die_roll, win_indicator, masked_actions = \
                before_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index,
                                         self.interface, self.die_roll)
        terminal = 0 if win_indicator == 0 else 1
        if done_indicator == 1:
            terminal = 1

        reward = self.reward_cal(win_indicator, action_num, masked_actions_reward)

        if terminal:
            if win_indicator == 1:
                terminal = 2
        state_space = self.interface.board_to_state(game_elements)

        self.game_elements = game_elements_ori
        self.num_active_players = num_active_players_ori
        # self.num_die_rolls = num_die_rolls_ori
        self.current_player_index = current_player_index_ori
        self.done_indicator = done_indicator_ori
        self.win_indicator = win_indicator_ori

        return state_space, reward, terminal, masked_actions#can put KG in info

    def seed(self, seed=0):
        np_random, seed1 = seeding.np_random(seed)
        self.seeds = seeding.hash_seed(seed1 + 1) % 2 ** 31
        # self.seeds = seed
        return self.seeds

    def save_kg(self):
        self.kg.build_kg_file(self.log_path, level='rel', use_hash=True)
        self.kg_save_num += 1
        #save knowledge graph when simulating num of games is self.kg_save_interval
        # if self.kg_save_num % self.kg_save_interval == 0:
        #     self.kg.save_json(self.kg.kg_rel, self.kg_rel_path)
        #     self.kg.build_matrix_dict()
        #     self.kg.sparse_matrix = self.kg.dict_to_matrix()
        #     self.kg.save_matrix()
            # self.kg.save_vector()

        # Visulalize the novelty change in the network
        if self.kg.kg_change != self.kg_change:
            file = open(self.rule_change_path, "a")
            file.write(str(self.kg.kg_change) + ' \n')
            file.close()

        self.kg_change = self.kg.kg_change[:]

    def save_gameboard(self, path=None):
        if path:
            return write_out_current_state_to_file(self.game_elements, self.upper_path + path)
        # else:
        #     return write_out_current_state_to_file(self.game_elements, self.saved_gameboard_path)

    def background_agent_decision(self, masked_actions):  # only consider 2 actions
        player = self.game_elements['players'][0]
        # if masked_actions[0] == 1:
        #     allowable_moves = {action_choices.buy_property, action_choices.concluded_actions}
        # else:
        #     return 1
        # actions = self.player_decision_agents['player_2'].make_post_roll_move(player, self.game_elements, allowable_moves, 1)
        # if actions[-1] == dict():
        #     return 1
        # else:
        #     return 0

        if masked_actions[0] == 1:
            asset = self.game_elements['location_sequence'][player.current_position]
            if type(asset) == RealEstateLocation or type(asset) == UtilityLocation or type(asset) == RailroadLocation:
                return 0 if self.player_decision_agents['player_2'].make_buy_property_decision(player, self.game_elements, asset) else 1
        return 1
