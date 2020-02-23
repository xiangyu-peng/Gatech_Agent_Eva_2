import sys
sys.path.append('/media/becky/GNOME-p3/monopoly_simulator')
from gameplay_simple_becky_v1 import *
from interface import Interface
import simple_background_agent_becky_v1
from gym.utils import seeding

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

    def init(self):
        self.num_active_players = 2
        self.num_die_rolls = 0
        self.current_player_index = 0
        self.a = Interface()
        self.params = dict()
        self.player_decision_agents = dict()
        self.reward = 0
        self.terminal = False
        self.player_decision_agents = dict()


    def seed(self, seed=None):
        np_random, seed1 = seeding.np_random(seed)
        self.seeds = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return self.seeds

    def reset(self):
        self.init()
        player_list = ['player_' + str(i + 1) for i in range(self.num_active_players)]
        for player_name in player_list:
            self.player_decision_agents[player_name] = simple_background_agent_becky_v1.decision_agent_methods
        self.game_elements = set_up_board('/media/becky/GNOME/monopoly_game_schema_v1-2.json', self.player_decision_agents, self.num_active_players)
        np.random.seed(self.seeds) #control the seed!!!!
        self.game_elements['seed'] = self.seeds
        self.game_elements['card_seed'] = self.seeds
        self.game_elements['choice_function'] = np.random.choice
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params = \
            before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)

        return self.a.board_to_state(self.game_elements) #use interface to get the vector state space
    def reward_cal(self):
        reward = self.game_elements['players'][self.current_player_index].current_cash
        return reward
    def next(self, action):
        if self.terminal:
            self.reset()
        action = self.a.action_num2vec(action)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index = \
            after_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, action, self.a,
                        self.params)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index = \
            simulate_game_step(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index)
        self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a, self.params = \
            before_agent(self.game_elements, self.num_active_players, self.num_die_rolls, self.current_player_index, self.a)
        self.reward = self.reward_cal()
        self.terminal = False if self.num_active_players > 1 else True
        return self.a.state_space, self.reward, self.terminal, self.a.masked_actions #can put KG in info