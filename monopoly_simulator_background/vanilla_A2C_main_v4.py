'''
This version include representation vector of game rule in the game state
Only consider 2 actions
Only take one action each time
Add hyperopt.
Add Tensorboard => python -m tensorboard.main --logdir=./Result --host 143.215.128.115
'''
import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation/GNOME-p3')
####################

from monopoly_simulator_background.vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from monopoly_simulator_background.gameplay_tf import *
import logger
from itertools import product
import argparse
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
class Config:
    device = torch.device('cuda:0')

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
    def __init__(self, params, device_id, gameboard=None, kg_use=True):
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
        self.kg_vector = []
        self.save_path = self.params['save_path']
        self.num_test = self.params['num_test']
        self.interface = Interface()
        self.gameboard = gameboard
        self.kg_use = kg_use

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

            #######################################

        with HiddenPrints():
            self.envs = ParallelEnv(self.n_train_processes, self.gameboard, self.kg_use)

        self.config_model = Config()
        self.config_model.hidden_state = self.params['hidden_state']
        self.config_model.action_space = self.params['action_space']
        self.config_model.state_num = self.params['state_num']

        if self.gameboard:
            self.interface.set_board(self.gameboard)
            self.config_model.state_num = len(self.interface.board_to_state(self.gameboard))

        self.model = ActorCritic(self.config_model)  # A2C model
        self.model.to(self.device)
        self.loss = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # plot the results
        self.test_result = []
        self.writer = SummaryWriter('./Result')

    def train_kg(self):
        with HiddenPrints():
            self.envs.reset()
        train_kg_num = 0
        while train_kg_num < 100000:
            train_kg_num += 1
            a_tf = [0 for i in range(self.n_train_processes)]
            s_prime, _, done, masked_actions = self.envs.step_after_nochange(a_tf)
            if masked_actions[0][1] != []:
                self.kg_vector = masked_actions[0][1]
                # print('self.kg_vector ===>',self.kg_vector)
                break

    def add_vector_to_state(self, states, vector, device):
        state_return = []
        for state in states:
            new_state = np.concatenate((vector.reshape(1, -1), [state]), axis=1)
            state_return.append(new_state[0])
        return torch.tensor(state_return, device=device).float()

    def save(self, step_idx):
        save_name = upper_path+ self.save_path + '/v4_lr_' + str(self.learning_rate) + '_#_' +  str(int(step_idx / self.PRINT_INTERVAL)) + '.pkl'
        torch.save(self.model, save_name)

    def plot_test(self):
        step_x = []
        loss_y = []
        winning_rate_y = []
        avg_score_y = []
        avg_diff_y = []
        for i in self.test_result:
            step_x.append(i['step'])
            loss_y.append(i['loss'].cpu().numpy().tolist())
            winning_rate_y.append(i['winning_rate'])
            avg_score_y.append(i['avg_score'])
            avg_diff_y.append(i['avg_diff'])
        data_x = [step_x for i in range(4)]
        data_y = [loss_y, winning_rate_y, avg_score_y, avg_diff_y]


        fig, axes = plt.subplots(figsize=(7,7), nrows=2, ncols=2, sharey=True, sharex=True)
        for i, ax in enumerate(axes.flatten()):
            ax.plot(data_x[i], data_y[i])
        fig.savefig('lr_' + str(self.learning_rate) + 'train_A2C.png')



    def train(self):
        # self.train_kg()
        self.kg_vector = np.load('/media/becky/GNOME-p3/KG_rule/vector.npy')
        step_idx = 1
        loss_train_min = [-1, 10]
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
                s = self.add_vector_to_state(s, self.kg_vector, self.device)
                prob = self.model.actor(s)  # s => tensor #output = prob for actions
                # prob = model.actor(torch.from_numpy(s,device=device).float()) # s => tensor #output = prob for actions
                # use the action from the distribution if it is in masked

                a = []
                for i in range(self.n_train_processes):
                    # action_Invalid = True
                    # num_loop = 0
                    a_once = Categorical(prob).sample().cpu().numpy()[i]  # substitute
                    # while action_Invalid:
                    #     a_once = Categorical(prob).sample().cpu().numpy()[i] #substitute
                    #     action_Invalid = True if masked_actions[i][a_once] == 0 else False
                    #     num_loop += 1

                    # if num_loop > 5:
                    #     a_once = largest_prob(prob[i], masked_actions)
                    #     break
                    a.append(a_once)
                # while opposite happens. Step won't change env, step_nochange changes the env\

                s_prime_cal, r, done, masked_actions = self.envs.step_nochange(a)

                values.append(self.model.critic(s))

                log_prob = Categorical(prob).log_prob(torch.tensor(a, device=self.device))
                entropy += Categorical(prob).entropy().mean()
                log_probs.append(log_prob)
                rewards.append(torch.FloatTensor(r).unsqueeze(1).to(self.device))
                done = [[0] if i > 0 else [1] for i in done]
                masks.append(torch.tensor(done, device=self.device).float())

                a_tf = [0 for i in range(self.n_train_processes)]
                s_prime, _, done_now, masked_actions = self.envs.step_after_nochange(a_tf)

                s = s_prime

                ##########
                s_prime_cal = self.add_vector_to_state(s_prime_cal, self.kg_vector, self.device)

                # loss cal
                log_probs = torch.cat(log_probs)
                returns = compute_returns(self.model.critic(s_prime_cal), rewards, masks, gamma=0.99)
                returns = torch.cat(returns).detach()
                values = torch.cat(values)
                advantage = returns - values

                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                loss += actor_loss + 0.5 * critic_loss - 0.001 * entropy



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
                    loss_train_min[0] = step_idx // self.PRINT_INTERVAL
                    loss_train_min[1] = loss_train / self.PRINT_INTERVAL
                self.TB.logkv('step_idx', step_idx)
                self.TB.logkv('loss_train', round(float(loss_train) / self.PRINT_INTERVAL, 3))
                loss_train = torch.tensor(0, device=self.device).float()
                # if step_idx % PRINT_INTERVAL == 0:
                avg_score, avg_winning, avg_diff = self.test_v2(step_idx, self.model, self.device,
                                                                num_test=self.num_test, kg_vector=self.kg_vector,
                                                                gameboard=self.gameboard, kg_use=self.kg_use)

                # Add test result to storage and then plot
                self.test_result.append({'step':step_idx // self.PRINT_INTERVAL, 'loss': loss_train / self.PRINT_INTERVAL,\
                        'winning_rate':avg_winning, 'avg_score':avg_score, 'avg_diff':avg_diff})

                self.TB.logkv('avg_score', avg_score)
                self.TB.logkv('avg_winning', avg_winning)
                self.TB.logkv('avg_diff', avg_diff)
                self.TB.dumpkvs()
                # save weights of A2C
                if step_idx % self.PRINT_INTERVAL == 0:
                    self.save(step_idx)
                print(self.model.named_parameters())
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if 'bn' not in name:
                        self.writer.add_histogram(name, param, 0)
                        self.writer.add_scalar('loss', loss_train / self.PRINT_INTERVAL, i)


            step_idx += 1
            # print('step_idx',step_idx)
        self.envs.close()
        return loss_train_min

    def test_v2(self,step_idx, model, device, num_test, kg_vector, gameboard, kg_use):
        # Set the env for testing
        env = gym.make('monopoly_simple-v1')
        env.set_kg(kg_use)
        env.set_board(gameboard)

        score = 0.0
        done = False
        win_num = 0
        avg_diff = 0
        with HiddenPrints():
            s, masked_actions = env.reset()
        for _ in range(num_test):
            num_game = 0
            score_game = 0

            while not done:
                num_game += 1
                s = s.reshape(1, -1)
                s = self.add_vector_to_state(s, kg_vector, device)
                prob = model.actor(s)
                a = Categorical(prob).sample().cpu().numpy()  # substitute
                with HiddenPrints():
                    s_prime, r, done, masked_actions = env.step(a[0])
                s = s_prime
                score_game += r
            avg_diff += s[-2] - s[-1]
            score += score_game/num_game + 10 * abs(abs(int(done) - 2) - 1)
            win_num += abs(abs(int(done) - 2) - 1)
            done = 0

        print(f"Step # :{step_idx}, avg score : {score/num_test:.3f}")
        print(f"Step # :{step_idx}, avg winning : {win_num / num_test:.3f}")
        print(f"Step # :{step_idx}, avg diff : {avg_diff / num_test:.3f}")
        env.close()
        return round(score/num_test, 5), round(win_num/num_test, 5), round(avg_diff/num_test, 5)

# Package: hyperopt- tune hyper-parameter
def hyper_func(params):
    trainer = MonopolyTrainer(params, device_id)
    loss_train_min = trainer.train()
    ret = {
        "loss": loss_train_min[1],
        "attachments": {
            "lr": params['learning_rate'],
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


    def dict_product(d):
        param_list = []
        keys = d.keys()
        for element in product(*d.values()):
            param_list.append(dict(zip(keys, element)))
        return param_list




    # specify device
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, required=False)
    parser.add_argument("--train_type", type=str, default='train', required=True) #train or hyperopt
    args = parser.parse_args()
    device_id = args.device




    if args.train_type == 'hyperopt':
        for key in all_params.keys():  # explore space
            all_params[key] = hp.choice(key, list(all_params[key]))

        trials = Trials()
        best = fmin(hyper_func,
                    space=all_params,
                    algo=tpe.suggest,
                    max_evals=2,
                    trials=trials)
        print('best:',best)
        best_params = space_eval(all_params, best)# at the point best using the hyperopt.space_eval function
                                                  #to see param_dict's parameter values in the output
        print('best_params:',best_params)
        trial_loss = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_loss)
        best_loss = -trial_loss[best_ind]
        best_lr = trials.trial_attachments(trials.trials[best_ind])["lr"]
        print('best_x:',best_lr)

        #plot the figure and save
        print('trials:')
        for trial in trials.trials[:2]:
            print(trial)
        f, ax = plt.subplots(1)
        xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
        ys = [t['result']['loss'] for t in trials.trials]
        ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
        ax.set_title('$loss$ $vs$ $learning_rate$ ', fontsize=18)
        ax.set_xlabel('$lr$', fontsize=16)
        ax.set_ylabel('$loss$', fontsize=16)
        f.savefig("Th_A2C.png")

    elif args.train_type == 'train':
        param_list = dict_product(all_params)
        for params in param_list:  # run different hyper parameters in sequence
            trainer = MonopolyTrainer(params, device_id)
            trainer.train()
            trainer.plot_test()  # Plot the training process
    else:
        print('Do Nothing -- Wrong code for train_type')
