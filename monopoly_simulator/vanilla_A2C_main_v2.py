from vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from gameplay_simple_tf import *

class Config:
    device = torch.device('cuda:3')
import os, sys

class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def largest_prob(prob, masked_actions):
    prob = prob.cpu().detach().numpy().reshape(-1,)
    # print('prob', prob)
    #Check if the action is valid
    action_Invalid = True
    largest_num = -1
    while action_Invalid:
        a = prob.argsort()[largest_num:][0]
        action_Invalid = True if masked_actions[a] == 0 else False
        largest_num -= 1
    return a

if __name__ == '__main__':
    config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
    config_data = ConfigParser()
    config_data.read(config_file)
    # print('config_data.items', config_data.sections())
    # Hyperparameters
    n_train_processes = 3
    learning_rate = 0.0002
    update_interval = 5
    gamma = 0.98
    max_train_steps = 60000
    PRINT_INTERVAL = 100
    config = Config()
    config.hidden_state = 256
    config.action_space = 2
    config.state_num = 56
    actor_loss_coefficient = 1
    save_dir = '/media/becky/GNOME-p3/monopoly_simulator'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")
    save_name = '/push_buy'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    #######################################
    with HiddenPrints():
        envs = ParallelEnv(n_train_processes)
    model = ActorCritic(config) #A2C model
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0

    with HiddenPrints():
        reset_array = envs.reset()
        s, masked_actions = [reset_array[i][0] for i in range(len(reset_array))], \
                            [reset_array[i][1] for i in range(len(reset_array))]


    # print('s',s)
    # print('mask', masked_actions)

        # s = torch.tensor(s, device=device).float()
        # masked_actions = torch.tensor(masked_actions, device=device)
    loss_train = torch.tensor(0, device=device).float()
    while step_idx < max_train_steps:
        loss = torch.tensor(0, device=device).float()
        for _ in range(update_interval):
            #move form last layer
            entropy = 0
            log_probs, masks, rewards, values = [], [], [], []

            # s = s.reshape(n_train_processes, -1)
            prob = model.actor(torch.tensor(s, device=device).float())  # s => tensor #output = prob for actions
            # prob = model.actor(torch.from_numpy(s,device=device).float()) # s => tensor #output = prob for actions
            #use the action from the distribution if it is in masked

            a = []
            for i in range(n_train_processes):
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
            #while opposite happens. Step won't change env, step_nochange changes the env\
            s_prime_cal, r, done, masked_actions = envs.step_nochange(a)

            values.append(model.critic(torch.tensor(s, device=device).float()))

            log_prob = Categorical(prob).log_prob(torch.tensor(a, device=device))
            entropy += Categorical(prob).entropy().mean()
            log_probs.append(log_prob)
            rewards.append(torch.FloatTensor(r).unsqueeze(1).to(device))
            done = [[1] if i > 0 else [0] for i in done]
            masks.append(torch.tensor(done, device=device).float())


            a_tf = [0 for i in range(n_train_processes)]
            s_prime, _, done, masked_actions = envs.step_after_nochange(a_tf)
            s = s_prime


            ##########
            s_prime_cal = torch.tensor(s_prime_cal, device=device).float()

            # loss cal
            log_probs = torch.cat(log_probs)
            returns = compute_returns(model.critic(s_prime_cal), rewards, masks, gamma=0.99)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss += actor_loss + 0.5 * critic_loss - 0.001 * entropy


        loss /= update_interval
        loss_train += loss
        # print('loss', loss)
        if loss != torch.tensor(0, device = device):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # graphviz.Source(make_dot(loss, params=dict(model.named_parameters()))).render('full_net')
        # print('weight after test = > ', model.fc_actor.weight)
        if step_idx % 100 == 0:
            print('loss_train ===>', loss_train / 100)
            loss_train = torch.tensor(0, device=device).float()
        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model,device, num_test=10)
            #save weights of A2C
            if step_idx % PRINT_INTERVAL == 0:
                save_path = '/media/becky/GNOME-p3/monopoly_simulator/weights'
                save_name = save_path + '/push_buy_tf_ne_' + str(int(step_idx / PRINT_INTERVAL)) + '.pkl'

                torch.save(model, save_name)

        step_idx += 1

    envs.close()