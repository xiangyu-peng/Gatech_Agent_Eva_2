# Without knowledge graph development
# Only consider 2 actions
# Only take one action each time
# log name + save name
# python vanilla_A2C_main_v3.py --novelty_change_num 18 --novelty_change_begin 1 --novelty_introduce_begin 0 --seed 8 --gat_use False

import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/monopoly_simulator','')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
sys.path.append(upper_path + '/KG_rule')
####################
from monopoly_simulator_background.vanilla_A2C import *
from monopoly_simulator_background.interface import Interface
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent

from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from monopoly_simulator_background.gameplay_tf import *
from monopoly_simulator_background import logger
from itertools import product
import argparse
import warnings
warnings.filterwarnings('ignore')
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class Config:
    device = torch.device('cuda:1')

class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# def largest_prob(prob, masked_actions):
#     prob = prob.cpu().detach().numpy().reshape(-1,)
#     # print('prob', prob)
#     #Check if the action is valid
#     action_Invalid = True
#     largest_num = -1
#     while action_Invalid:
#         a = prob.argsort()[largest_num:][0]
#         action_Invalid = True if masked_actions[a] == 0 else False
#         largest_num -= 1
#     return a
class MonopolyTrainer:
    def __init__(self,
                 params,
                 gameboard=None,
                 exp_dict=None):

        self.exp_dict = exp_dict
        self.gat_use = args.gat_use
        self.seed = args.seed

        #tf use
        self.tf_use = args.tf_use
        self.tf_stop_num = args.tf_stop_num

        self.config_file = args.config_file
        self._device_id = 'cuda:' + args.device_id
        self.params = params
        self.PRINT_INTERVAL = self.params['print_interval']
        self.n_train_processes = self.params['n_train_processes']  # batch size
        self.learning_rate = self.params['learning_rate']
        self.update_interval = self.params['update_interval']
        self.gamma = self.params['gamma']
        self.max_train_steps = self.params['max_train_steps']
        self.hidden_state = self.params['hidden_state']
        self.action_space = self.params['action_space']
        self.state_num = self.params['state_num']
        self.actor_loss_coefficient = self.params['actor_loss_coefficient']
        self.kg_vector = []
        self.interface = Interface()
        self.save_path = self.params['save_path']
        self.num_test = self.params['num_test']
        self.gameboard = gameboard
        self.kg_use = args.kg_use
        self.logger_use = args.logger_use
        self.test_result = {'step': [], 'loss': [],
                     'winning_rate': [], 'avg_score': [], 'avg_diff': []}
        self.spend_time = 0
        self.test_required = args.test_required


        # config logger
        if self.logger_use:
            hyper_config_str = '_n' + str(self.n_train_processes) + '_lr' + str(self.learning_rate) + '_ui' + str(
                self.update_interval) \
                               + '_y' + str(self.gamma) + '_s' + str(self.max_train_steps) + '_hs' + str(self.hidden_state) \
                               + '_as' + str(self.action_space) + '_sn' \
                               + '_ac' + str(self.actor_loss_coefficient) + '_seed' + str(self.seed) + '_novelty_' + \
                               str(exp_dict['novelty_num'][0]) + '_' + str(exp_dict['novelty_num'][1]) + '_ran'
            self.TB = logger.Logger(None, [logger.make_output_format('csv', 'logs/', log_suffix=hyper_config_str)])

        self.start_time = datetime.datetime.now()

        ###set to cpu for evaluation
        self.device = torch.device(self._device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
            #######################################

        with HiddenPrints():
            self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use, self.config_file, self.seed, self.exp_dict)

        if self.gameboard:
            self.interface.set_board(self.gameboard)
        else:
            self.interface.set_board('/monopoly_simulator_background/baseline_interface.json')

        if args.pretrain_model:
            self.model = torch.load(args.pretrain_model, map_location={"cuda:2" : "cuda:1"})
        else:

            self.config_model = ConfigParser()
            self.config_model.hidden_state = self.params['hidden_state']
            self.config_model.action_space = self.params['action_space']
            self.config_model.state_num = self.params['state_num']
            # self.config_model.state_output_size = self.params['state_output_size']

            if self.gameboard:
                self.config_model.state_num = len(self.interface.board_to_state(self.gameboard))
            self.model = ActorCritic(self.config_model, args.gat_use)  # A2C model

        self.model.to(self.device)
        self.loss = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.memory = Memory()

    def add_vector_to_state(self, states, vector, device):
        state_return = []
        for state in states:
            new_state = np.concatenate((vector.reshape(1, -1), [state]), axis=1)
            state_return.append(new_state[0])
        return torch.tensor(state_return, device=device).float()

    def test_v2(self, step_idx, seed, config_file=None, num_test=1000):

        # model_path = '/media/becky/GNOME-p3/monopoly_simulator_background/weights/v3_lr_0.001_#_' + str(step_idx) + '.pkl'
        # model = torch.load(model_path)

        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        env.set_config_file(config_file)
        env.set_exp(self.exp_dict)
        env.set_kg(self.kg_use)
        env.set_board()  # self.gameboard
        env.seed(seed)
        score = 0.0
        done = False
        win_num = 0
        avg_diff = 0

        with HiddenPrints():
            s, masked_actions = env.reset()

        for _ in range(num_test):
            num_game = 0
            score_game = 0
            # print('reset', s, masked_actions)
            done_type = True
            while not done:
                # print(s, masked_actions)
                num_game += 1
                s = s.reshape(1, -1)
                s = torch.tensor(s, device=self.device).float()
                prob = self.model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                # print(a[0])
                if masked_actions[a[0]] == 0:
                    a = [1]
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])
                s = s_prime

                if done_type:
                    # print('reset', s, masked_actions)
                    done_type = False

                score_game += r

            avg_diff += s[-2] - s[-1]
            score += score_game/num_game  + 1 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0

        if self.logger_use:
            print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
            print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
            print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")

            end_time = datetime.datetime.now()
            self.spend_time = (end_time - self.start_time).seconds / 60 / 60  # hr
            # print('step => ', step_idx // self.PRINT_INTERVAL, ' costs ',
            #       (end_time - self.start_time).seconds / 60, ' mins')

        env.close()

        return round(score/num_test, 5), round(win_num/num_test, 5), round(avg_diff/num_test, 5)

    def save(self, step_idx):
        save_name = self.save_path + '/del_ran_19_1_v3_lr_' + str(self.learning_rate) + '_#_' +  str(int(step_idx / self.PRINT_INTERVAL)) + '.pkl'
        torch.save(self.model, save_name)

        # for i, (name, param) in enumerate(self.model.named_parameters()):
        #     if i == 0:
        #         print('output', param)

        # self.model = torch.load(save_name)
        # self.model.to(self.device)
        #
        # self.loss = 0
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # print('optimizer', self.optimizer)

    def set_gameboard(self, gameboard=None,
                      seed=None,
                      save_path=None,
                      test_required=None,
                      print_interval=None,
                      max_train_steps=None,
                      logger_use=None,
                      logger_name=None):

        self.gameboard = gameboard if gameboard else self.gameboard
        self.save_path = save_path if save_path else self.save_path
        self.seed = seed if seed else self.seed
        self.test_required = test_required
        self.PRINT_INTERVAL = print_interval if print_interval else self.PRINT_INTERVAL
        self.max_train_steps = max_train_steps if max_train_steps else self.max_train_steps
        if logger_use == True:
            self.logger_use = True
            hyper_config_str = '_n' + str(self.n_train_processes) + '_lr' + str(self.learning_rate) + '_ui' + str(
                self.update_interval) \
                               + '_y' + str(self.gamma) + '_s' + str(self.max_train_steps) + '_hs' + str(self.hidden_state) \
                               + '_as' + str(self.action_space) + '_sn'  \
                               + '_ac' + str(self.actor_loss_coefficient) + '_hyp_' + logger_name

            self.TB = logger.Logger(None, [logger.make_output_format('csv', 'logs/', log_suffix=hyper_config_str)])

    def train(self):
        # with HiddenPrints():
        #     self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use, self.config_file, self.seed)

        step_idx = 1
        with HiddenPrints():
            reset_array = self.envs.reset()

            s, masked_actions, background_actions = [reset_array[i][0] for i in range(len(reset_array))], \
                                [reset_array[i][1][0] for i in range(len(reset_array))],\
                                [reset_array[i][1][1] for i in range(len(reset_array))]

        loss_train = torch.tensor(0, device=self.device).float()
        while step_idx < self.max_train_steps and self.spend_time < 300:

            loss = torch.tensor(0, device=self.device).float()
            for _ in range(self.update_interval):
                entropy = 0
                log_probs, masks, rewards, values = [], [], [], []

                # s = s.reshape(n_train_processes, -1)
                prob = self.model.actor(torch.tensor(s, device=self.device).float())  # s => tensor #output = prob for actions
                # prob = model.actor(torch.from_numpy(s,device=device).float()) # s => tensor #output = prob for actions
                # use the action from the distribution if it is in masked
                # print(prob, s)
                a = []
                for i in range(self.n_train_processes):
                    a_once = Categorical(prob).sample().cpu().numpy()[i]  # substitute
                    if masked_actions[i][a_once] == 0:
                        a_once = 1
                    a.append(a_once)

                # while opposite happens. Step won't change env, step_nochange changes the env\
                # if self.tf_use:
                s_prime_cal, r, done, info = self.envs.step_nochange(a)

                # if self.interface.check_relative_state(s[0], self.novelty_spaces) and a[0] == 0:
                #     print(s[0].tolist()[-52:-12].index(1))
                #     print(s[0].tolist()[11])
                #     print(a[0])
                #     print(r[0])
                #     print(s[0].tolist()[-12:].index(1), s_prime_cal[0].tolist()[-12:].index(1))
                #     print(s[0].tolist()[-6:].index(1), s_prime_cal[0].tolist()[-6:].index(1))
                #     print(s_prime_cal[0].tolist()[11])

                # else:
                #     s_prime_cal, r, done, info = self.envs.step(a)
                # s_prime_cal = torch.tensor(s_prime_cal, device=self.device).float()

                values.append(self.model.critic(torch.tensor(s, device=self.device).float()))

                log_prob = Categorical(prob).log_prob(torch.tensor(a, device=self.device))
                entropy += Categorical(prob).entropy().mean()
                log_probs.append(log_prob)
                rewards.append(torch.FloatTensor(r).unsqueeze(1).to(self.device))

                done = [[0] if i > 0 else [1] for i in done]
                masks.append(torch.tensor(done, device=self.device).float())

                # Teacher forcing part #
                # a_tf = []
                # for i in range(self.n_train_processes):
                #     if masked_actions[i][0] == 1:
                #         a_tf.append(0)
                #     else:
                #         a_tf.append(1)
                if self.tf_use:
                    a_tf = [1 for i in range(self.n_train_processes)]
                    if step_idx < self.tf_stop_num: #background_actions
                        s_prime, _, _, info = self.envs.step_after_nochange(background_actions)  # background_actions
                    else:
                        s_prime, _, _, info = self.envs.step_after_nochange(a)


                else:
                    # s = s_prime_cal
                    s_prime, _, _, info = self.envs.step_after_nochange(a)
                s = s_prime
                #########################
                masked_actions = [info_[0] for info_ in info]  # Update masked actions
                background_actions = [info_[1] for info_ in info]

                ##########
                s_prime_cal = torch.tensor(s_prime_cal, device=self.device).float()

                # loss cal
                log_probs = torch.cat(log_probs)
                returns = compute_returns(self.model.critic(s_prime_cal), rewards, masks, gamma=0.99)
                # print('rewards', rewards)
                # print(self.model.critic(s_prime_cal))

                returns = torch.cat(returns).detach()
                # print('returns',returns)
                values = torch.cat(values)
                advantage = returns - values
                # print('Advan',rewards, advantage, a, background_actions, s)
                # print('log_probs',log_probs)
                actor_loss = -(log_probs * advantage.detach()).mean()
                # print('actor_loss',actor_loss)
                critic_loss = advantage.pow(2).mean()
                # print('critic_loss',critic_loss)
                loss_once = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                if loss_once != torch.tensor(float('nan'), device=self.device):
                    loss += loss_once
                self.memory.clear()
                # print('loss', loss)

            loss /= self.update_interval
            self.loss = loss
            loss_train += loss

            if loss != torch.tensor(0, device=self.device):
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                # print(self.optimizer.step())


            # if step_idx % 10 == 0:
            #     for i, (name, param) in enumerate(self.model.named_parameters()):
            #         if i == 0:
            #             print('input', param)

            avg_score, avg_winning, avg_diff = 0, 0, 0
            if step_idx % self.PRINT_INTERVAL == 0:
                # save weights of A2C
                self.save(step_idx)

                # print('loss', loss_train / self.PRINT_INTERVAL, prob[0])  #, s[0].tolist()[-12-40:-12].index(1), s[0].tolist()[-12: -6].index(1)*500)
                loss_return  = loss_train / self.PRINT_INTERVAL
                loss_return = loss_train.cpu().detach().numpy()
                loss_train = torch.tensor(0, device=self.device).float()

                if self.test_required:
                    avg_score, avg_winning, avg_diff = self.test_v2(step_idx, seed=0, config_file=self.config_file)
                    print(avg_score, avg_winning, avg_diff)
                    # Add test result to storage and then plot
                    self.test_result['step'].append(step_idx // self.PRINT_INTERVAL)
                    self.test_result['loss'].append(loss_train.cpu().detach().numpy() / self.PRINT_INTERVAL)
                    self.test_result['winning_rate'].append(avg_winning)
                    self.test_result['avg_score'].append(avg_score)
                    self.test_result['avg_diff'].append(avg_diff)

                if self.logger_use:
                    self.TB.logkv('step_idx', step_idx)
                    self.TB.logkv('loss_train', round(float(loss_train) / self.PRINT_INTERVAL, 3))
                    if self.test_required:
                        self.TB.logkv('avg_score', avg_score)
                        self.TB.logkv('avg_winning', avg_winning)
                        self.TB.logkv('avg_diff', avg_diff)
                    self.TB.dumpkvs()

                    # plot the results
                    # print(self.model.named_parameters())
                    # for i, (name, param) in enumerate(self.model.named_parameters()):
                    #     if 'bn' not in name:
                    #         self.writer.add_histogram(name, param, 0)
                    #         self.writer.add_scalar('loss', loss_train / self.PRINT_INTERVAL, i)


            step_idx += 1
            end_time = datetime.datetime.now()

            self.spend_time = (end_time - self.start_time).seconds / 60 / 60

            # # After converge, we will stop the training
            # if self.test_required:
            #     if len(self.test_result['winning_rate']) > 50:
            #         if max(self.test_result['winning_rate'][:-20]) > max(self.test_result['winning_rate'][-20:]) +  0.05:
            #             break
            #         if (max(self.test_result['winning_rate'][-15:]) - min(self.test_result['winning_rate'][-15:])) < 0.03:
            #             break

        self.envs.close()

        return loss_return, avg_score, avg_winning, avg_diff

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        default='/monopoly_simulator_background/config.ini',
                        required=False,
                        help="config_file for env")
    parser.add_argument('--novelty_change_num', type=int,
                        default=None, required=True,
                        help="Novelty price change number")
    parser.add_argument('--novelty_change_begin', type=int,
                        default=None, required=True,
                        help="Novelty price change begin number")
    parser.add_argument('--novelty_introduce_begin', type=int,
                        default=0, required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--seed', type=int,
                        default=0, required=True,
                        help="Env seed")
    parser.add_argument('--upper_path', type=str,
                        default='/media/becky/GNOME-p3', required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--exp_type', type=str,
                        default='kg', required=False,
                        help="Novelty price change begin number")
    parser.add_argument('--device_id',
                        default='1', type=str,
                        required=False,
                        help="GPU id we use")
    parser.add_argument('--tf_use',
                        default=True, type=bool,
                        required=False,
                        help="Use tf or not")
    parser.add_argument('--tf_stop_num',
                        default=20000, type=int,
                        required=False,
                        help="When to stop using tf")
    parser.add_argument('--test_required',
                        default=True, type=bool,
                        required=False,
                        help="We need to run tests or not")
    parser.add_argument('--kg_use',
                        default=False, type=bool,
                        required=False,
                        help="Use kg to learn the game rule change or not")
    parser.add_argument('--logger_use',
                        default=True, type=bool,
                        required=False,
                        help="Use csv file to record the test results or not")
    parser.add_argument('--pretrain_model',
                        default=None,
                        required=False,
                        help="pretrain model path")
    parser.add_argument('--gat_use',
                        default=False, type=bool,
                        required=True,
                        help="Whether use the kg graph attention. Yes: 'kg'; No: False")

    args = parser.parse_args()
    args.gat_use = False if args.gat_use != 'kg' else args.gat_use

    exp_dict = dict()
    exp_dict['novelty_num'] = (args.novelty_change_num, args.novelty_change_begin)
    exp_dict['novelty_inject_num'] = args.novelty_introduce_begin
    exp_dict['exp_type'] = args.exp_type

    # read the config file and set the hyper-param
    config_data = ConfigParser()
    config_data.read(args.upper_path + args.config_file)

    # set param_list: a list of dictionary
    all_params = {}
    # print('config_data',config_data['hyper'])
    for key in config_data['hyper']:
        v = eval(config_data['hyper'][key])
        if not isinstance(v, tuple):
            all_params[key] = (v,)
        else:
            all_params[key] = v

    def dict_product(d):
        param_list = []
        keys = d.keys()
        for element in product(*d.values()):
            param_list.append(dict(zip(keys, element)))
        return param_list

    param_list = dict_product(all_params)
    for params in param_list:  # run different hyper parameters in sequence
        trainer = MonopolyTrainer(params,
                                  gameboard=None,
                                  exp_dict=exp_dict)  # '/media/becky/GNOME-p3/monopoly_simulator_background/weights/no_v3_lr_0.0001_#_107.pkl'
    trainer.train()

# if __name__ == '__main__':
#     config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
#     config_data = ConfigParser()
#     config_data.read(config_file)
#     # print('config_data.items', config_data.sections())
#     # Hyperparameters
#     n_train_processes = 1
#     learning_rate = 0.0002
#     update_interval = 5
#     gamma = 0.98
#     max_train_steps = 60000
#     PRINT_INTERVAL = 100
#     config = Config()
#     config.hidden_state = 256
#     config.action_space = 2
#     config.state_num = 56
#     actor_loss_coefficient = 1
#     save_dir = '/media/becky/GNOME-p3/monopoly_simulator'
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:1" if use_cuda else "cpu")
#     save_name = '/push_buy'
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#
#     #######################################
#     with HiddenPrints():
#         envs = ParallelEnv(n_train_processes)
#     model = ActorCritic(config) #A2C model
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     step_idx = 0
#
#     with HiddenPrints():
#         reset_array = envs.reset()
#         s, masked_actions = [reset_array[i][0] for i in range(len(reset_array))], \
#                             [reset_array[i][1] for i in range(len(reset_array))]
#
#     # print('s',s)
#     # print('mask', masked_actions)
#
#         # s = torch.tensor(s, device=device).float()
#         # masked_actions = torch.tensor(masked_actions, device=device)
#     loss_train = torch.tensor(0, device=device).float()
#     while step_idx < max_train_steps:
#         loss = torch.tensor(0, device=device).float()
#         for _ in range(update_interval):
#             #move form last layer
#             entropy = 0
#             log_probs, masks, rewards, values = [], [], [], []
#
#             # s = s.reshape(n_train_processes, -1)
#             prob = model.actor(torch.tensor(s, device=device).float())  # s => tensor #output = prob for actions
#             # prob = model.actor(torch.from_numpy(s,device=device).float()) # s => tensor #output = prob for actions
#             #use the action from the distribution if it is in masked
#
#             a = []
#             for i in range(n_train_processes):
#                 # action_Invalid = True
#                 # num_loop = 0
#                 a_once = Categorical(prob).sample().cpu().numpy()[i]  # substitute
#                 # while action_Invalid:
#                 #     a_once = Categorical(prob).sample().cpu().numpy()[i] #substitute
#                 #     action_Invalid = True if masked_actions[i][a_once] == 0 else False
#                 #     num_loop += 1
#
#                     # if num_loop > 5:
#                     #     a_once = largest_prob(prob[i], masked_actions)
#                     #     break
#                 a.append(a_once)
#             #while opposite happens. Step won't change env, step_nochange changes the env\
#             s_prime_cal, r, done, masked_actions = envs.step_nochange(a)
#
#             values.append(model.critic(torch.tensor(s, device=device).float()))
#
#             log_prob = Categorical(prob).log_prob(torch.tensor(a, device=device))
#             entropy += Categorical(prob).entropy().mean()
#             log_probs.append(log_prob)
#             rewards.append(torch.FloatTensor(r).unsqueeze(1).to(device))
#             done = [[1] if i > 0 else [0] for i in done]
#             masks.append(torch.tensor(done, device=device).float())
#
#
#             a_tf = [0 for i in range(n_train_processes)]
#             s_prime, _, done, masked_actions = envs.step_after_nochange(a_tf)
#             s = s_prime
#
#
#             ##########
#             s_prime_cal = torch.tensor(s_prime_cal, device=device).float()
#
#             # loss cal
#             log_probs = torch.cat(log_probs)
#             returns = compute_returns(model.critic(s_prime_cal), rewards, masks, gamma=0.99)
#             returns = torch.cat(returns).detach()
#             values = torch.cat(values)
#             advantage = returns - values
#
#             actor_loss = -(log_probs * advantage.detach()).mean()
#             critic_loss = advantage.pow(2).mean()
#
#             loss += actor_loss + 0.5 * critic_loss - 0.001 * entropy
#
#
#         loss /= update_interval
#         loss_train += loss
#         # print('loss', loss)
#         if loss != torch.tensor(0, device = device):
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # graphviz.Source(make_dot(loss, params=dict(model.named_parameters()))).render('full_net')
#         # print('weight after test = > ', model.fc_actor.weight)
#         if step_idx % 100 == 0:
#             print('loss_train ===>', loss_train / 100)
#             loss_train = torch.tensor(0, device=device).float()
#         if step_idx % PRINT_INTERVAL == 0:
#             test(step_idx, model,device, num_test=10)
#             #save weights of A2C
#             if step_idx % PRINT_INTERVAL == 0:
#                 save_path = '/media/becky/GNOME-p3/monopoly_simulator/weights'
#                 save_name = save_path + '/push_buy_tf_ne_' + str(int(step_idx / PRINT_INTERVAL)) + '.pkl'
#
#                 torch.save(model, save_name)
#
#         step_idx += 1
#
#     envs.close()
