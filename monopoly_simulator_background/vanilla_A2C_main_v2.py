# Without knowledge graph development
# Only consider 2 actions
# Only take one action each time
# no TF!
# Add hyperopt

from vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from gameplay_tf import *
import logger
import os
import sys
from itertools import product
import argparse
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
warnings.filterwarnings('ignore')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# class Config:
#     device = torch.device('cuda:3')

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
    def __init__(self, params, device_id):
        self._device_id = device_id
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
        self.save_path = self.params['save_path']  # add hyperopt
        self.kg_vector = []

        # config logger
        hyper_config_str = '_n' + str(self.n_train_processes) + '_lr' + str(self.learning_rate) + '_ui' + str(
            self.update_interval) \
                           + '_y' + str(self.gamma) + '_s' + str(self.max_train_steps) + '_hs' + str(self.hidden_state) \
                           + '_as' + str(self.action_space) + '_sn' + str(self.state_num) \
                           + '_ac' + str(self.actor_loss_coefficient)
        self.TB = logger.Logger(None, [logger.make_output_format('csv', 'logs/', log_suffix=hyper_config_str)])

        if not self._device_id:  # use all available devices
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
            self.device = torch.device('cuda:0')

        ###set to cpu for evaluation
        self.device = torch.device('cpu')
            #######################################

        with HiddenPrints():
            self.envs = ParallelEnv(self.n_train_processes)

        self.config_model = ConfigParser()
        self.config_model.hidden_state = self.params['hidden_state']
        self.config_model.action_space = self.params['action_space']
        self.config_model.state_num = self.params['state_num']

        self.model = ActorCritic(self.config_model)  # A2C model
        self.model.to(self.device)
        self.loss = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.memory = Memory()

    def save(self, step_idx):
        save_name = self.save_path + 'v2_lr_' + str(self.learning_rate) + '_#_' +  str(int(step_idx / self.PRINT_INTERVAL)) + '.pkl'
        torch.save(self.model, save_name)

    def train(self):
        step_idx = 1
        loss_train_min = [-1,10]

        with HiddenPrints():
            reset_array = self.envs.reset()
            s, masked_actions = [reset_array[i][0] for i in range(len(reset_array))], \
                                [reset_array[i][1] for i in range(len(reset_array))]

        loss_train = torch.tensor(0, device=self.device).float()
        while step_idx < self.max_train_steps:
            loss = torch.tensor(0, device=self.device).float()
            for _ in range(self.update_interval):
                # move form last layer
                entropy = 0
                log_probs, masks, rewards, values = [], [], [], []

                prob = self.model.actor(torch.tensor(s, device=self.device).float())  # s => tensor #output = prob for actions

                a = []
                for i in range(self.n_train_processes):
                    a_once = Categorical(prob).sample().cpu().numpy()[i]  # substitute
                    a.append(a_once)
                # while opposite happens. Step won't change env, step_nochange changes the env\
                s_prime_cal, r, done, _ = self.envs.step(a)
                # s_prime_cal = torch.tensor(s_prime_cal, device=self.device).float()

                values.append(self.model.critic(torch.tensor(s, device=self.device).float()))

                log_prob = Categorical(prob).log_prob(torch.tensor(a, device=self.device))
                entropy += Categorical(prob).entropy().mean()
                log_probs.append(log_prob)
                rewards.append(torch.FloatTensor(r).unsqueeze(1).to(self.device))
                # print('done', done)
                done = [[0] if i > 0 else [1] for i in done]
                masks.append(torch.tensor(done, device=self.device).float())

                s = s_prime_cal

                ##########
                s_prime_cal = torch.tensor(s_prime_cal, device=self.device).float()

                # loss cal
                log_probs = torch.cat(log_probs)
                returns = compute_returns(self.model.critic(s_prime_cal), rewards, masks, gamma=0.99)

                returns = torch.cat(returns).detach()
                # print('returns',returns)
                values = torch.cat(values)
                advantage = returns - values
                # print('log_probs',log_probs)
                actor_loss = -(log_probs * advantage.detach()).mean()
                # print('actor_loss',actor_loss)
                critic_loss = advantage.pow(2).mean()
                # print('critic_loss',critic_loss)
                loss += actor_loss + 0.5 * critic_loss - 0.001 * entropy
                self.memory.clear()
                # print('loss',loss)

            loss /= self.update_interval
            self.loss = loss
            loss_train += loss
            # print('loss', loss)
            if loss != torch.tensor(0, device=self.device):
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            if step_idx % self.PRINT_INTERVAL == 0:
                print('loss_train ===>', loss_train / self.PRINT_INTERVAL)
                if loss_train_min[-1] > (loss_train / self.PRINT_INTERVAL):
                    loss_train_min[0] = step_idx % self.PRINT_INTERVAL
                    loss_train_min[1] = loss_train / self.PRINT_INTERVAL

                self.TB.logkv('step_idx', step_idx)
                self.TB.logkv('loss_train', round(float(loss_train) / self.PRINT_INTERVAL, 3))
                loss_train = torch.tensor(0, device=self.device).float()
                # if step_idx % PRINT_INTERVAL == 0:
                avg_score, avg_winnning = test(step_idx, self.model, self.device, num_test=1)
                self.TB.logkv('avg_score', avg_score)
                self.TB.logkv('avg_winning', avg_winnning)
                self.TB.dumpkvs()
                # save weights of A2C
                if step_idx % self.PRINT_INTERVAL == 0:
                    self.save(step_idx)  # add hyperopt

            step_idx += 1
            # print('step_idx',step_idx)
        self.envs.close()
        return loss_train_min

def dict_product(d):
    param_list = []
    keys = d.keys()
    for element in product(*d.values()):
        param_list.append(dict(zip(keys, element)))
    return param_list

def hyper_func(params):
    trainer = MonopolyTrainer(params, device_id)
    loss_train_min = trainer.train()
    ret = {
        "loss": loss,
        "attachments": {
            "x": params['learning_rate'],
            'step':loss_train_min[0]
        },
        "status": STATUS_OK
    }
    return ret

if __name__ == '__main__':
    # read the config file and set the hyper-param
    config_file = 'config.ini'
    config_data = ConfigParser()
    config_data.read(config_file)

    # set param_list: a list of dictionary
    all_params = {}
    # print('config_data',config_data['hyper'])
    for key in config_data['hyper']:
        v = eval(config_data['hyper'][key])
        if not isinstance(v, tuple):
            all_params[key] = (v,)
        else:
            all_params[key] = v

    param_list = dict_product(all_params)

    # specify device
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    device_id = args.device

    for key in all_params.keys():  # explore space
        all_params[key] = hp.choice(key, list(all_params[key]))

    # for params in param_list:  # run different hyper parameters in sequence
    #     trainer = MonopolyTrainer(params, device_id)
    #     trainer.train()

    trials = Trials()
    best = fmin(hyper_func, all_params, tpe.suggest, 10, trials)
    print('best:',best)
    best_params = space_eval(all_params, best)# at the point best using the hyperopt.space_eval function
                                              #to see param_dict's parameter values in the output
    print('best_params:',best_params)
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_loss)
    best_loss = -trial_loss[best_ind]
    best_x = trials.trial_attachments(trials.trials[best_ind])["x"]
    print('best_x:',best_x)
