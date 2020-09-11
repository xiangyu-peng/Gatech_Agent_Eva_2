import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator','')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
sys.path.append(upper_path + '/KG_rule')
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import numpy as np
import os, sys
from random import randint
from GNN.model import *
from KG_rule.kg_GAT import RGCNetwork
from GNN.RGCN.model import RGCN
from KG_rule.partially_kg import Adj_Gen
from torch import FloatTensor
import copy

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#A2C Model
class ActorCritic(nn.Module):
    def __init__(self, config, device, gat_use=False):
        super(ActorCritic, self).__init__()
        torch.cuda.set_device(device)
        self.device=device
        self.config = config
        self.gat_use = gat_use
        self.fc_actor = nn.Linear(config.hidden_state, config.action_space) #config.action_space= 2; hidden_size = 256
        self.fc_critic = nn.Linear(config.hidden_state, 1)
        if gat_use:
            if gat_use == 'state':
                self.fc_1 = nn.Linear(config.gat_output_size + config.state_output_size,
                                      config.hidden_state)  # config.state_num = 4; config.hidden_state) = 256
                self.state_gat = GraphNN(config.gat_emb_size,
                                               config.embedding_size,
                                               config.dropout_ratio,
                                               config.entity_id_file_path_state,
                                               config.state_output_size,
                                               'state',
                                               device).to(device)
                self.graph_gat = GraphNN(config.gat_emb_size,
                                         config.embedding_size,
                                         config.dropout_ratio,
                                         config.entity_id_file_path,
                                         config.gat_output_size,
                                         'kg',
                                         device).to(device)
                # self.state_nn = StateNN(config.state_num, config.state_output_size)

            elif gat_use == 'gcn':
                self.graph_gat = GraphNN(config.gat_emb_size,
                                         config.embedding_size,
                                         config.dropout_ratio,
                                         config.entity_id_file_path,
                                         config.gat_output_size,
                                         'kg',
                                         device).to(device)
                self.fc_1 = nn.Linear(config.gat_output_size + config.state_num,
                                      config.hidden_state)
                with open(config.entity_id_file_path_state, 'rb')as f:
                    feature_dict = pickle.load(f)

                # self.state_graph = RGCNetwork(config.state_num,
                #                               config.hidden_dim_gcn,
                #                               config.dropout_ratio,
                #                               support=1,
                #                               num_bases=-1,
                #                               device=device,
                #                               output_size=1,
                #                               feature_dict=feature_dict).cuda()

                self.state_graph = RGCN(i_dim=config.state_num,
                                        h_dim=config.hidden_dim_gcn,
                                        drop_prob=0,
                                        support=4,
                                        num_bases=1,
                                        device=device).to(device)
            else:
                self.fc_1 = nn.Linear(config.state_output_size + config.gat_output_size,
                                      config.hidden_state)  # config.state_num = 4; config.hidden_state) = 256
                self.graph_gat = GraphNN(config.gat_emb_size,
                                              config.embedding_size,
                                              config.dropout_ratio,
                                              config.entity_id_file_path,
                                              config.gat_output_size,
                                              'kg').to(device) #.cuda()
                self.state_nn = StateNN(config.state_num, config.state_output_size)
                self.adj_gen = Adj_Gen()
        else:
            self.fc_1 = nn.Linear(config.state_num,
                                  config.hidden_state)  # config.state_num = 4; config.hidden_state) = 256
            # self.state_nn = StateNN(config.state_num, config.state_output_size)

    #Actor model
    def actor(self, x, softmax_dim=1): #actions' probability
        x = F.relu(self.fc_1(x))
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def critic(self, x):
        x = F.relu(self.fc_1(x))
        v = self.fc_critic(x)
        return v

    def novelty_detect(self, statewitha):
        x = F.relu(self.fc_2(statewitha))
        label = self.detector(x)
        return label

    def forward_baseline(self, state):
        return state
        # return self.state_nn.forward(state)
        # state = self.state_nn.forward(state)
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # g_t = torch.tensor(np.ones([1,20]) * 0.01, device=self.device).type(dtype)
        # g_t_cat = g_t
        # for i in range(state.shape[0] - 1):
        #     g_t_cat = torch.cat((g_t_cat, g_t), dim=0)
        # return torch.cat((g_t_cat, state), dim=1)

    def forward(self, state, adj):
        position = self.adj_gen.get_pos(state)  # state -> position
        state = self.state_nn.forward(state)
        g_t_cat = FloatTensor().to(self.device)
        for n in range(len(position)):
            adj_part = self.adj_gen.output_part_adj(adj, position[n])  # adj -> partial adj
            adj_part = torch.IntTensor(adj_part).to(self.device)
            g_t = self.graph_gat.forward(adj_part).reshape(1, -1)
            g_t_cat = torch.cat((g_t_cat, g_t), dim=0)
        return torch.cat((g_t_cat, state), dim=1)

    def forward_state(self, state, adj):
        o_t = self.state_gat.forward(state)
        o_t = o_t.reshape(len(state), -1) * 100
        g_t = self.graph_gat.forward(adj).reshape(1, -1)
        g_t_cat = g_t
        for i in range(len(state) -  1):
            g_t_cat = torch.cat((g_t_cat, g_t), dim=0)
        return torch.cat((g_t_cat, o_t), dim=1)

    def forward_gcn(self, state, adj):
        # print('weight', self.graph_gat.fc1.weight[0])
        # print(self.state_graph.gc2.W)
        # print(self.graph_gat.fc1.grad)
        o_t = self.state_graph.forward(state)
        # o_t = o_t.reshape(len(state), -1)
        g_t = self.graph_gat.forward(adj).reshape(1, -1)
        g_t_cat = g_t
        o_t_cat = o_t[0].reshape(1, -1) * 10
        for i in range(1, len(state)):
            g_t_cat = torch.cat((g_t_cat, g_t), dim=0)
            o_t_cat = torch.cat((o_t_cat, o_t[i].reshape(1, -1) * 10), dim=0)
        return torch.cat((g_t_cat, o_t_cat), dim=1)


def worker(worker_id, master_end, worker_end, gameboard=None, kg_use=True, config_file=None, seed=0, exp_dict=None):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make('monopoly_simple-v1')
    env.set_config_file(config_file)
    env.set_exp(exp_dict)
    if worker_id > 0:
        env.set_kg(False)
    else:
        env.set_kg(kg_use)
    env.set_board(gameboard)
    env.seed(seed + worker_id)
    # env.seed(randint(0,sys.maxsize))

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, masked_actions = env.reset()
            worker_end.send((ob, masked_actions))
        elif cmd == 'reset_task':
            ob, masked_actions = env.reset_task()
            worker_end.send((ob, masked_actions))
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        elif cmd == 'step_nochange':
            ob, reward, done, info = env.step_nochange(data)
            worker_end.send((ob, reward, done, info))
        elif cmd == 'step_after_nochange':
            ob, reward, done, info = env.step_after_nochange(data)
            worker_end.send((ob, reward, done, info))
        elif cmd == 'step_hyp':
            ob, reward, done, info = env.step_hyp(data)
            worker_end.send((ob, reward, done, info))
        elif cmd == 'output_novelty':
            if worker_id > 0:
                worker_end.send(None)
            else:
                kg_change = copy.deepcopy(env.output_kg_change())
                worker_end.send(kg_change)
                # worker_end.send(env.output_kg())
        elif cmd == 'set_exp':
            env.set_exp(data)
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, gameboard=None, kg_use=True, config_file=None, seed=0, exp_dict=None):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)]) #pipe connect each other
        self.master_ends, self.worker_ends = master_ends, worker_ends


        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end,gameboard, kg_use, config_file, seed, exp_dict))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def set_exp(self, exp_dict):
        for master_end in self.master_ends:
            master_end.send(('set_exp', exp_dict))

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_async_nochange(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step_nochange', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_async_after_nochange(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step_after_nochange', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_async_hyp(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step_hyp', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends] #receive from worker_end #format???
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))   #send to worker_end =>
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async(actions)
        return self.step_wait()

    def step_nochange(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async_nochange(actions)
        return self.step_wait()
    def step_after_nochange(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async_after_nochange(actions)
        return self.step_wait()

    def step_hyp(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async_hyp(actions)
        return self.step_wait()

    def output_novelty(self):
        for master_end in self.master_ends:
            master_end.send(('output_novelty', None))
            return master_end.recv()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def test(step_idx, model, device, num_test, gameboard=None,kg_use=False, config_file=None, exp_dict=None):
    env = gym.make('monopoly_simple-v1')
    env.set_config_file(config_file)
    env.set_dict(exp_dict)
    env.set_kg(kg_use)
    env.set_board(gameboard)

    score = 0.0
    done = False
    win_num = 0
    for _ in range(num_test):
        with HiddenPrints():
            s, masked_actions = env.reset()
        num_game = 0
        score_game = 0

        while not done:
            num_game += 1
            s = s.reshape(1, -1)
            prob = model.actor(torch.tensor(s, device=device).float())

            # Choose the action with highest prob and not in masked action
            # Becky#########################################################
            prob = prob.cpu().detach().numpy().reshape(-1, )
            # if num_game == 15:
            #     print(prob)
            # Check if the action is valid
            action_Invalid = True
            largest_num = -1
            while action_Invalid:
                a = prob.argsort()[largest_num:][0]
                action_Invalid = True if masked_actions[a] == 0 else False
                largest_num -= 1

                # a = Categorical(prob).sample().cpu().numpy()  # substitute
                # action_Invalid = True if masked_actions[a[0]] == 0 else False
            #
            # a = Categorical(prob).sample().numpy()
            with HiddenPrints():
                s_prime, r, done, masked_actions = env.step(a)
            # masked_actions = masked_actions[0]
            # print(a)
            # s_prime, r, done, info = env.step(a)
            s = s_prime
            score_game += r
        # print(s)
        score += score_game/num_game
        win_num += int(done) - 1
        done = 0
    # print('weight = test> ', model.fc_actor.weight)
    print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
    print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
    env.close()
    return round(score/num_test, 3), round(win_num/num_test, 3)


def compute_target(v_final, r_lst, mask_lst, gamma): #may update

    G = v_final.reshape(-1) #i.e. [0.13882717]
    td_target = list()
    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

def compute_returns(final_value, rewards, mask_lst, gamma=0.99):
    R = final_value
    # print('final_value',final_value)
    returns = []
    for step in reversed(range(len(rewards))):
        # print(step, 'rewards[step]', rewards[step],'mask_lst[step]', mask_lst[step] )
        R = rewards[step] + gamma * R * mask_lst[step]

        returns.insert(0, R)
    return returns


# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)

# import gym
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical
# import torch.multiprocessing as mp
# import time
# import numpy as np
# import os, sys
# from random import randint
#
# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
#
# #A2C Model
# class ActorCritic(nn.Module):
#     def __init__(self,config):
#         super(ActorCritic, self).__init__()
#         self.config = config
#         self.fc_1 = nn.Linear(config.state_num, config.hidden_state) #config.state_num = 4; config.hidden_state) = 256
#         self.fc_actor = nn.Linear(config.hidden_state, config.action_space) #config.action_space= 2; hidden_size = 256
#         self.fc_critic = nn.Linear(config.hidden_state, 1)
#
#         #add rnn herenvidia-smi
#
#
#     #Actor model
#     def actor(self, x, softmax_dim=1): #actions' probability
#         x = F.relu(self.fc_1(x))
#         x = self.fc_actor(x)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob
#
#     def critic(self, x):
#         x = F.relu(self.fc_1(x))
#         v = self.fc_critic(x)
#         return v
#
# def worker(worker_id, master_end, worker_end):
#     master_end.close()  # Forbid worker to use the master end for messaging
#     env = gym.make('monopoly_simple-v1')
#     env.seed(worker_id)
#     # env.seed(randint(0,sys.maxsize))
#
#     while True:
#         cmd, data = worker_end.recv()
#         if cmd == 'step':
#             ob, reward, done, info = env.step(data)
#             worker_end.send((ob, reward, done, info))
#         elif cmd == 'reset':
#             ob, masked_actions = env.reset()
#             worker_end.send((ob, masked_actions))
#         elif cmd == 'reset_task':
#             ob, masked_actions = env.reset_task()
#             worker_end.send((ob, masked_actions))
#         elif cmd == 'close':
#             worker_end.close()
#             break
#         elif cmd == 'get_spaces':
#             worker_end.send((env.observation_space, env.action_space))
#         elif cmd == 'step_nochange':
#             ob, reward, done, info = env.step_nochange(data)
#             worker_end.send((ob, reward, done, info))
#         elif cmd == 'step_after_nochange':
#             ob, reward, done, info = env.step_after_nochange(data)
#             worker_end.send((ob, reward, done, info))
#         else:
#             raise NotImplementedError
#
# class ParallelEnv:
#     def __init__(self, n_train_processes):
#         self.nenvs = n_train_processes
#         self.waiting = False
#         self.closed = False
#         self.workers = list()
#
#         master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)]) #pipe connect each other
#         self.master_ends, self.worker_ends = master_ends, worker_ends
#
#
#         for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
#             p = mp.Process(target=worker,
#                            args=(worker_id, master_end, worker_end))
#             p.daemon = True
#             p.start()
#             self.workers.append(p)
#
#         # Forbid master to use the worker end for messaging
#         for worker_end in worker_ends:
#             worker_end.close()
#
#     def step_async(self, actions):
#         for master_end, action in zip(self.master_ends, actions):
#             master_end.send(('step', action)) #send to worker_end => 'step' & action
#         self.waiting = True  #waiting???
#
#     def step_async_nochange(self, actions):
#         for master_end, action in zip(self.master_ends, actions):
#             master_end.send(('step_nochange', action)) #send to worker_end => 'step' & action
#         self.waiting = True  #waiting???
#
#     def step_async_after_nochange(self, actions):
#         for master_end, action in zip(self.master_ends, actions):
#             master_end.send(('step_after_nochange', action)) #send to worker_end => 'step' & action
#         self.waiting = True  #waiting???
#
#     def step_wait(self):
#         results = [master_end.recv() for master_end in self.master_ends] #receive from worker_end #format???
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rews), np.stack(dones), infos
#
#     def reset(self):
#         for master_end in self.master_ends:
#             master_end.send(('reset', None))   #send to worker_end =>
#         return np.stack([master_end.recv() for master_end in self.master_ends])
#
#     def step(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
#         self.step_async(actions)
#         return self.step_wait()
#
#     def step_nochange(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
#         self.step_async_nochange(actions)
#         return self.step_wait()
#     def step_after_nochange(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
#         self.step_async_after_nochange(actions)
#         return self.step_wait()
#
#     def close(self):  # For clean up resources
#         if self.closed:
#             return
#         if self.waiting:
#             [master_end.recv() for master_end in self.master_ends]
#         for master_end in self.master_ends:
#             master_end.send(('close', None))
#         for worker in self.workers:
#             worker.join()
#             self.closed = True
#
# def test(step_idx, model, device, num_test):
#     env = gym.make('monopoly_simple-v1')
#     score = 0.0
#     done = False
#     win_num = 0
#     for _ in range(num_test):
#         with HiddenPrints():
#             s, masked_actions = env.reset()
#         num_game = 0
#         score_game = 0
#
#         while not done:
#             num_game += 1
#             s = s.reshape(1, -1)
#             prob = model.actor(torch.tensor(s, device=device).float())
#
#             # Choose the action with highest prob and not in masked action
#             # Becky#########################################################
#             prob = prob.cpu().detach().numpy().reshape(-1, )
#             # if num_game == 15:
#             #     print(prob)
#             # Check if the action is valid
#             action_Invalid = True
#             largest_num = -1
#             while action_Invalid:
#                 a = prob.argsort()[largest_num:][0]
#                 action_Invalid = True if masked_actions[a] == 0 else False
#                 largest_num -= 1
#
#                 # a = Categorical(prob).sample().cpu().numpy()  # substitute
#                 # action_Invalid = True if masked_actions[a[0]] == 0 else False
#             #
#             # a = Categorical(prob).sample().numpy()
#             with HiddenPrints():
#                 s_prime, r, done, masked_actions = env.step(a)
#             # masked_actions = masked_actions[0]
#             # print(a)
#             # s_prime, r, done, info = env.step(a)
#             s = s_prime
#             score_game += r
#         # print(s)
#         score += score_game/num_game
#         win_num += int(done) - 1
#         done = 0
#     # print('weight = test> ', model.fc_actor.weight)
#     print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
#     print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
#
#     env.close()
#
# def compute_target(v_final, r_lst, mask_lst, gamma): #may update
#
#     G = v_final.reshape(-1) #i.e. [0.13882717]
#     td_target = list()
#     for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
#         G = r + gamma * G * mask
#         td_target.append(G)
#
#     return torch.tensor(td_target[::-1]).float()
#
# def compute_returns(final_value, rewards, mask_lst, gamma=0.99):
#     R = final_value
#     # print('final_value',final_value)
#     returns = []
#     for step in reversed(range(len(rewards))):
#         # print(step, 'rewards[step]', rewards[step],'mask_lst[step]', mask_lst[step] )
#         R = rewards[step] + gamma * R * mask_lst[step]
#
#         returns.insert(0, R)
#     return returns
#
# # def compute_returns(next_value, rewards, masks, gamma=0.99):
# #     R = next_value
# #     returns = []
# #     for step in reversed(range(len(rewards))):
# #         R = rewards[step] + gamma * R * masks[step]
# #         returns.insert(0, R)
# #     return returns
#
#
# # if __name__ == '__main__':
#     # envs = ParallelEnv(n_train_processes) #The simulator environment
#     #
#     # model = ActorCritic(None) #A2C model
#     # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     #
#     # step_idx = 0
#     # s = envs.reset() #initilize s
#     # while step_idx < max_train_steps:
#     #     s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
#     #     for _ in range(update_interval):
#     #         prob = model.pi(torch.from_numpy(s).float())
#     #         a = Categorical(prob).sample().numpy()
#     #         s_prime, r, done, info = envs.step(a)
#     #
#     #         s_lst.append(s)
#     #         a_lst.append(a)
#     #         r_lst.append(r/100.0)
#     #         mask_lst.append(1 - done)
#     #
#     #         s = s_prime
#     #         step_idx += 1
#     #
#     #     s_final = torch.from_numpy(s_prime).float()
#     #     v_final = model.v(s_final).detach().clone().numpy()
#     #     td_target = compute_target(v_final, r_lst, mask_lst)
#     #
#     #     td_target_vec = td_target.reshape(-1)
#     #     s_vec = torch.tensor(s_lst).float().reshape(-1, 4)  # 4 == Dimension of state
#     #     a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
#     #     advantage = td_target_vec - model.v(s_vec).reshape(-1)
#     #
#     #     pi = model.pi(s_vec, softmax_dim=1)
#     #     pi_a = pi.gather(1, a_vec).reshape(-1)
#     #     loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
#     #         F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)
#     #
#     #     optimizer.zero_grad()
#     #     loss.backward()
#     #     optimizer.step()
#     #
#     #     if step_idx % PRINT_INTERVAL == 0:
#     #         test(step_idx, model)
#     # # print('s_lst', s_lst)
#     # envs.close()

class Novelty_detect(nn.Module):
    def __init__(self, config, device):
        super(Novelty_detect, self).__init__()
        torch.cuda.set_device(device)
        self.device=device
        self.config = config
        self.fc_2 = nn.Linear(config.state_num + 1 - 40, config.hidden_state)
        self.detector = nn.Linear(config.hidden_state, 1)

    def forward(self, statewitha):
        x = F.relu(self.fc_2(statewitha))
        label = self.detector(x)
        return label