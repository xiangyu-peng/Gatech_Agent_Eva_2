import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import sys
sys.path.append('/mnt/c/Users/spenc/Documents/2021Projects/DARPA-SAILON/Gatech_Agent_Eva_2/env')
from monopoly_world import Monopoly_world
# from lm.next_state_provider import HuggingFaceNextStateProvider
# from summary_provider.dummy_summary import DummySummaryProvider
# from summary_provider.individual_sentiment_summary import IndividualSentimentSummary
# from data_utils.data_provider import GutenburgDataProvider


try:
    from debug_print import print
except:
    pass
import numpy as np

from gym.envs.registration import register
import gym

def self_register():
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'monopoly_simple-v1' in env:
            print('Remove {} from registry'.format(env))
            del gym.envs.registration.registry.env_specs[env]
    register(
        id='monopoly_simple-v1',
        entry_point='gym_simulator_env.envs:Sim_Monopoly',
    )

class Sim_Monopoly(gym.Env):
    """
    To run this env, U must follow the remaining procedure!
    env = gym.make('monopoly_simple-v1')
    env.set_config_file()
    env.set_kg(False)
    env.set_board(path)
    env.seed(seed=1)

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.MonopolyWorld = None
        self.observation_space = []
        self.masked_actions = []
        self.seeds = 0
        self.reward = 0
        self.terminal = False
        self.info = None

    def step(self, action):
        self.observation_space, self.reward, self.terminal, self.info = self.MonopolyWorld.next(action)
        return self.observation_space, self.reward, self.terminal, self.info

    def step_nochange(self, action):
        observation_space, reward, terminal, info = self.MonopolyWorld.next_nochange(action)
        return observation_space, reward, terminal, info

    def step_after_nochange(self, action):
        observation_space, reward, terminal, info = self.MonopolyWorld.next_after_nochange(action)
        return observation_space, reward, terminal, info

    def step_hyp(self, action):
        observation_space, reward, terminal, info = self.MonopolyWorld.next_hyp(action)
        return observation_space, reward, terminal, info

    def reset(self):
        self.observation_space, self.masked_actions = self.MonopolyWorld.reset()
        return self.observation_space, self.masked_actions

    def seed(self, seed=None):
        self.seeds = self.MonopolyWorld.seed(seed)
        return self.seeds

    def set_board(self, board=None):
        self.MonopolyWorld.set_initial_gameboard(gameboard=board)

    def set_kg(self, kg_use):
        self.MonopolyWorld.kg_use = kg_use
    # def render(self, mode='human', close=False):
    #     pass

    def output_kg_change(self):
        return self.MonopolyWorld.kg_change[:]

    def output_interface(self):
        return self.MonopolyWorld.interface

    def save_gameboard(self, path=None):
        return self.MonopolyWorld.save_gameboard(path)

    def set_config_file(self, config_data=None):
        if config_data:
            self.MonopolyWorld = Monopoly_world(config_data)
        else:
            self.MonopolyWorld = Monopoly_world()
            
    def set_exp(self, exp_dict):
        self.MonopolyWorld.set_exp(exp_dict)

    def output_kg(self):
        return self.MonopolyWorld.kg.kg_rel
