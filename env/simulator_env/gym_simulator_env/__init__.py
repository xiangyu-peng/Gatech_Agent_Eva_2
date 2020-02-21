from gym.envs.registration import register
import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'monopoly_simple-v1' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registration.registry.env_specs[env]
register(
    id='monopoly_simple-v1',
    entry_point='gym_simulator_env.envs:Sim_Monopoly'
)