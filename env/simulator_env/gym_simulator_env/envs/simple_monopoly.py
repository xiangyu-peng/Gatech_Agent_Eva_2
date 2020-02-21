import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import sys
sys.path.append('/media/becky/GNOME-p3/env')
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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        world = Monopoly_world()
        self.MonopolyWorld = world
        self.observation_space = [0 for i in range(52 + 2 * self.MonopolyWorld.num_active_players)]
        self.action_space = [0 for i in range(79)]

    def step(self, action):
        next_state, reward, terminal, info = self.MonopolyWorld.next(action)
        return next_state, reward, terminal, info

    def reset(self):
        self.observation_space = self.MonopolyWorld.reset()
        return self.observation_space

    # def render(self, mode='human', close=False):
    #     pass