import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
####################
from monopoly_simulator_background.vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt
import random

class Config:
    device = torch.device('cuda:2')
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

def test_v2(step_idx, model, device, num_test, seed, config_file):
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if i == 0:
    #         print('test_out', param)

    # Set the env for testing
    env = gym.make('monopoly_simple-v1')
    env.set_config_file(config_file)
    env.set_kg(False)
    env.set_board()
    env.seed(seed)

    score = 0.0
    done = False
    win_num = 0
    avg_diff = 0
    pie_num = 0

    with HiddenPrints():
        s, (masked_actions, background_action) = env.reset()

    game_num = 0
    for _ in range(num_test):
        game_num += 1

        num_game = 0
        score_game = 0
        background_action_num = 0

        while not done:
            num_game += 1
            s = s.reshape(1, -1)
            s = torch.tensor(s, device=device).float()

            prob = model.actor(s)
            # break


            # if num_game == 2:
            #     print(prob, s)
            #     s= s[0]
            #     break
            a = Categorical(prob).sample().cpu().numpy()  # substitute
            # a = [prob.cpu().tolist()[0].index(max(prob.cpu().tolist()[0]))]
            if masked_actions[a[0]] == 0:
                a = [1]
            # print(masked_actions, a[0])
            # if num_game == 10:
            #     print('s', s[0][-6], masked_actions, s[0][-2], s)
            #     print(prob, a[0])
            #     break
            with HiddenPrints():
                a_random = random.randint(0,1)
                print(a_random)
                s_prime, r, done, (masked_actions,background_action) = env.step(a[0])
                # s_prime, r, done, (masked_actions,background_action) = env.step(background_action)
            if background_action == 0:
                background_action_num += 1
            # print(s_prime, done, masked_actions,background_action)

            s = s_prime
            score_game += r
        # s = s.cpu().numpy()[0]

        avg_diff += s[-2] - s[-1]
        score += score_game/num_game + 1 * abs(abs(int(done) - 2) - 1)
        win_num += abs(abs(int(done) - 2) - 1)
        if done ==3:
            pie_num += 1
        done = 0
        # print(background_action_num/num_game)

        # if game_num == num_test / 2 or game_num == num_test:
        #     print(f"Step # :{step_idx}, avg winning : {win_num / game_num:.3f}")
        #     print('========', num_test, game_num)


    print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
    print(f"Step # :{step_idx}, avg winning : {win_num / num_test/(1-pie_num / num_test):.3f}")
    print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")
    print(f"Step # :{step_idx}, avg pie : {pie_num / num_test:.3f}")

    env.close()

    return round(score/num_test, 5), round(win_num/num_test, 5), round(avg_diff/num_test, 5)

def test_eva(step_idx, model, device, num_test, vector):
    env = gym.make('monopoly_simple-v1')

    score = 0.0
    done = False
    win_num = 0
    skip_num = 0
    buy_num = 0
    else_num = 0
    diff_total_pos = 0
    diff_total_neg = 0
    game_num = 0
    for _ in range(num_test):
        game_num += 1

        with HiddenPrints():
            s, masked_actions = env.reset()
        num_game = 0
        score_game = 0
        while not done:
            s = s.reshape(1, -1)
            # print('s', s)
            # s = add_vector_to_state(s, vector, device)


            num_game += 1
            s = torch.tensor(s, device=device).float()
            before_action = model.critic(s)
            prob = model.actor(s)

            #debug
            # print('s',s)
            # print('prob', pr ob)
            # if num_game > 0:
            #     break

            # Choose the action with highest prob and not in masked action
            # Becky#########################################################
            # prob = prob.cpu().detach().numpy().reshape(-1, )


            # if num_game == 15:
            #     print(prob)
            # Check if the action is valid
            # action_Invalid = True
            # largest_num = -1
            # num_loop = 0
            # while action_Invalid:
            #     # print('prob', prob)
            #     a = prob.argsort()[largest_num:][0]
            #     action_Invalid = True if masked_actions[a] == 0 else False
            #     largest_num -= 1
            #     # a = Categorical(prob).sample().cpu().numpy()  # substitute
            #     # action_Invalid = True if masked_actions[a[0]] == 0 else False
            #     # num_loop += 1
            #     # if num_loop > 20:
            #     #     a = largest_prob(prob, masked_actions)
            #     #     break
            # #
            a = Categorical(prob).sample().cpu().numpy()
            a = a[0]
            if a == 79:
                skip_num += 1
            elif a == 0:
                buy_num += 1

            with HiddenPrints():

                s_prime, r, done, masked_actions = env.step(a)
            # if done:
                # print(s_prime)
            #debug
            # s_prime_cal = torch.tensor(s_prime, device=device).float()
            # print('after action',model.critic(s_prime_cal) - before_action)
            # break



            # print(a)
            # s_prime, r, done, info = env.step(a)
            s = s_prime
            score_game += r
        # print(s)
        score += score_game/num_game + 10 * abs(abs(int(done) - 2) - 1)
        if done == 3:
            else_num += 1
            diff_total_pos += max(0, s[-4] - min(s[-3:]))
            if s[-4] > max(s[-3:]):
                win_num += 1

        if done == 2:
            win_num += int(done) - 1
            diff_total_pos += s[-4]
        if done == 1:
            diff_total_neg += max(s[-3:])
        done = 0

        if game_num == num_test/2 or num_test:
            print(f"Step # :{step_idx}, avg winning : {win_num / game_num:.3f}")



        # print('s===>',s)
        # print('skip_num', skip_num, 'buy_num', buy_num, 'else_num', else_num)
    # print('weight = test> ', model.fc_actor.weight)

    print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
    print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
    print(f"Step # :{step_idx}, avg pie : {else_num / num_test:.3f}")
    print(f"Step # :{step_idx}, avg diff : {diff_total_pos / num_test:.3f}")
    print(f"Step # :{step_idx}, avg diff_neg : {diff_total_neg / num_test:.3f}")
    return score/num_test, win_num / num_test, else_num / num_test, diff_total_pos / num_test, diff_total_neg

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
    config.action_space = 86
    save_dir = '/media/becky/GNOME-p3/monopoly_simulator_background'
    # device = torch.device('cuda:0')
    save_name = '/push_buy'
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    #######################################
    # with HiddenPrints():
    #     envs = ParallelEnv(n_train_processes)
    # vector = np.load('/media/becky/GNOME-p3/KG-rule/vector.npy')
    score_list, win_list, pie_list, diff_list, diff_neg_list = [], [], [], [], []
    for seed in range(8,9):
        s_ran = [random.randint(-1, 1) for k in range(40)]
        # s_ran[11] = -1
        for i in range(2,25,1):  # From 0(1) to 6(2) and interval is 5(3) => [0,5]
            model_path = '/media/becky/GNOME-p3/monopoly_simulator_background/weights/hyp_1_7_seed8_v3_lr_0.0001_#_' +str(i) + '.pkl'
            # model_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights/hyp_1/hyp_v3_lr_0.0001_#_' +str(i) + '.pkl'
            # model_path = '/media/becky/GNOME-p3/Hypothetical_simulator/weights/hyp/60/_60/hyp_v3_lr_0.0001_#_1.pkl'
            # model_path = '/media/becky/GNOME-p3/monopoly_simulator_background/weights/no_v3_lr_0.0001_#_107.pkl'
            # model_path = '/media/becky/GNOME-p3/monopoly_simulator_background/weights/ran_2_7_v3_lr_0.0001_#_'+str(i) +'.pkl'

            model = torch.load(model_path, map_location={"cuda:2": "cuda:1"})
            score, win, diff = test_v2(i , model, device, 100, seed=seed, config_file='/Hypothetical_simulator/config_offline_hyp_1.ini')

            # for j in range(40):
            #     if j == 8:
            #         import random
            #         s = [0 for i in range(40)]
            #         s[j] = 1
            #         s_m = [0,0,0,1,0,0]
            #         import random
            #         s = s_ran + s + s_m + s_m
            #
            #         s = np.array(s)
            #         s = s.reshape(1, -1)
            #         s = torch.tensor(s, device=device).float()
            #
            #         prob = model.actor(s)
            #         print(j, prob)
            # score_list.append(score)
            # win_list.append(win)
            # pie_list.append(pie)
            # diff_list.append(diff)
            # diff_neg_list.append(diff_neg)


    # #plot the figure
    # x = np.linspace(0, 87, 30)
    # plt.figure()
    # plt.plot(x, score_list)
    # plt.plot(x, win_list)
    # plt.plot(x, pie_list)
    # plt.plot(x, diff_list)
    # # plt.plot(x, diff_neg_list)
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("Evaluate_A2C.png")