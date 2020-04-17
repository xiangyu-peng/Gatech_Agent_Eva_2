from vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
import numpy as np


class Config:
    device = torch.device('cuda:0')
import os, sys

class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def add_vector_to_state(state, vector, device):
    new_state = np.concatenate((vector.reshape(1, -1), state), axis=1)
    return torch.tensor(new_state, device=device).float()

def largest_prob(prob, masked_actions):
    prob = prob.cpu().detach().numpy().reshape(-1,)
    #Check if the action is valid
    action_Invalid = True
    largest_num = -1
    while action_Invalid:
        a = prob.argsort()[largest_num:][0]
        action_Invalid = True if masked_actions[a] == 0 else False
        a = [a]
        largest_num -= 1
    return a

def test_eva(step_idx, model, device, num_test, vector):
    env = gym.make('monopoly_simple-v1')
    score = 0.0
    done = False
    win_num = 0
    skip_num = 0
    buy_num = 0
    else_num = 0
    for _ in range(num_test):
        with HiddenPrints():
            s, masked_actions = env.reset()
        num_game = 0
        score_game = 0
        while not done:
            s = s.reshape(1, -1)
            s = add_vector_to_state(s, vector, device)
            num_game += 1
            # s = torch.tensor(s, device=device).float()
            before_action = model.critic(s)
            prob = model.actor(s)

            #debug
            print('s',s)
            print('prob', prob)
            if num_game > 0:
                break

            # Choose the action with highest prob and not in masked action
            # Becky#########################################################
            prob = prob.cpu().detach().numpy().reshape(-1, )


            # if num_game == 15:
            #     print(prob)
            # Check if the action is valid
            action_Invalid = True
            largest_num = -1
            num_loop = 0
            while action_Invalid:
                # print('prob', prob)
                a = prob.argsort()[largest_num:][0]
                action_Invalid = True if masked_actions[a] == 0 else False
                largest_num -= 1
                # a = Categorical(prob).sample().cpu().numpy()  # substitute
                # action_Invalid = True if masked_actions[a[0]] == 0 else False
                # num_loop += 1
                # if num_loop > 20:
                #     a = largest_prob(prob, masked_actions)
                #     break
            #
            # a = Categorical(prob).sample().numpy()
            # a = a[0]
            if a == 79:
                skip_num += 1
            elif a == 0:
                buy_num += 1
            with HiddenPrints():

                s_prime, r, done, masked_actions = env.step(a)

            #debug
            # s_prime_cal = torch.tensor(s_prime, device=device).float()
            # print('after action',model.critic(s_prime_cal) - before_action)
            # break
            
            
            
            # print(a)
            # s_prime, r, done, info = env.step(a)
            s = s_prime
            score_game += r
        # print(s)
        score += score_game/num_game
        win_num += int(done) - 1
        done = 0
        # print('s===>',s)
        # print('skip_num', skip_num, 'buy_num', buy_num, 'else_num', else_num)
    # print('weight = test> ', model.fc_actor.weight)
    print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
    print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")

if __name__ == '__main__':
    config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
    config_data = ConfigParser()
    config_data.read(config_file)
    # print('config_data.items', config_data.sections())
    # Hyperparameters
    n_train_processes = 1
    learning_rate = 0.0002
    update_interval = 5
    gamma = 0.98
    max_train_steps = 60000
    PRINT_INTERVAL = update_interval * 10
    config = Config()
    config.hidden_state = 256
    config.action_space = 2
    config.state_num = 216
    save_dir = '/media/becky/GNOME-p3/monopoly_simulator'
    device = torch.device('cuda:0')
    save_name = '/push_buy'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # torch.cuda.set_device(1)
    # device = torch.device("cuda", 1)

    #######################################
    with HiddenPrints():
        envs = ParallelEnv(n_train_processes)
    vector = np.load('/media/becky/GNOME-p3/KG-rule/vector.npy')
    # print(vector)
    for i in range(8):
        model_path = '/media/becky/GNOME-p3/monopoly_simulator/weights/push_buy_tf_ne_v4_' +str(i) + '.pkl'
        model = torch.load(model_path)
        print('i = ', i)
        test_eva(1, model, device, 1, vector)
    # i = 0
    # model_path = '/media/becky/GNOME-p3/monopoly_simulator/weights/push_buy_tf_ne_v4_' + str(i) + '.pkl'
    # model = torch.load(model_path)
    # test_eva(1, model, device, num_test=1000)