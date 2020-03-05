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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#A2C Model
class ActorCritic(nn.Module):
    def __init__(self,config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.fc_1 = nn.Linear(config.state_num, config.hidden_state) #config.state_num = 4; config.hidden_state) = 256
        self.fc_actor = nn.Linear(config.hidden_state, config.action_space) #config.action_space= 2; hidden_size = 256
        self.fc_critic = nn.Linear(config.hidden_state, 1)

        #add rnn herenvidia-smi


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

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make('monopoly_simple-v1')
    env.seed(worker_id)
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
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)]) #pipe connect each other
        self.master_ends, self.worker_ends = master_ends, worker_ends


        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            # worker_id = randint(0, sys.maxsize)
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_async_nochange(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step_nochange', action)) #send to worker_end => 'step' & action
        self.waiting = True  #waiting???

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends] #receive from worker_end #format???
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))   #send to worker_end =>
        return master_end.recv()

    def step(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async(actions)
        return self.step_wait()

    def step_nochange(self, actions):  #update actions => return np.stack(obs), np.stack(rews), np.stack(dones), infos
        self.step_async_nochange(actions)
        return self.step_wait()

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

def test(step_idx, model, device):
    env = gym.make('monopoly_simple-v1')
    score = 0.0
    done = False
    num_test = 10

    for _ in range(num_test):
        with HiddenPrints():
            s, masked_actions = env.reset()
        num_game = 0
        score_game = 0
        win_num = 0
        while not done:
            num_game += 1

            prob = model.actor(torch.tensor(s, device=device).float(), softmax_dim=0)

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
            #
            # a = Categorical(prob).sample().numpy()
            with HiddenPrints():
                s_prime, r, done, info = env.step(a)
            # s_prime, r, done, info = env.step(a)
            s = s_prime
            score_game += r
            # print(r)
        score += score_game/num_game
        win_num += done - 1
        done = False
    # print('weight = test> ', model.fc_actor.weight)
    print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
    print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.1f}")

    env.close()

def compute_target(v_final, r_lst, mask_lst, gamma): #may update

    G = v_final.reshape(-1) #i.e. [0.13882717]
    td_target = list()
    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# if __name__ == '__main__':
    # envs = ParallelEnv(n_train_processes) #The simulator environment
    #
    # model = ActorCritic(None) #A2C model
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #
    # step_idx = 0
    # s = envs.reset() #initilize s
    # while step_idx < max_train_steps:
    #     s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
    #     for _ in range(update_interval):
    #         prob = model.pi(torch.from_numpy(s).float())
    #         a = Categorical(prob).sample().numpy()
    #         s_prime, r, done, info = envs.step(a)
    #
    #         s_lst.append(s)
    #         a_lst.append(a)
    #         r_lst.append(r/100.0)
    #         mask_lst.append(1 - done)
    #
    #         s = s_prime
    #         step_idx += 1
    #
    #     s_final = torch.from_numpy(s_prime).float()
    #     v_final = model.v(s_final).detach().clone().numpy()
    #     td_target = compute_target(v_final, r_lst, mask_lst)
    #
    #     td_target_vec = td_target.reshape(-1)
    #     s_vec = torch.tensor(s_lst).float().reshape(-1, 4)  # 4 == Dimension of state
    #     a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
    #     advantage = td_target_vec - model.v(s_vec).reshape(-1)
    #
    #     pi = model.pi(s_vec, softmax_dim=1)
    #     pi_a = pi.gather(1, a_vec).reshape(-1)
    #     loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
    #         F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if step_idx % PRINT_INTERVAL == 0:
    #         test(step_idx, model)
    # # print('s_lst', s_lst)
    # envs.close()
