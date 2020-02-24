from vanilla_A2C import *
from configparser import ConfigParser


class Config:
    DEVICE = torch.device('cuda:0')
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
    config_data = ConfigParser()
    config_data.read(config_file)
    print('config_data.items', config_data.sections())
    # Hyperparameters
    n_train_processes = 1
    learning_rate = 0.0002
    update_interval = 5
    gamma = 0.98
    max_train_steps = 60000
    PRINT_INTERVAL = update_interval * 100
    config = Config()
    config.hidden_state = 256
    config.action_space = 80
    config.state_num = 56
    #######################################
    with HiddenPrints():
        envs = ParallelEnv(n_train_processes)
    model = ActorCritic(config) #A2C model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    with HiddenPrints():
        s, masked_actions = envs.reset()
    num = 0

    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list() #state list; action list, reward list, masked action list？？？

        for _ in range(update_interval): #substitute
        ########Becky###############################
        ##loop until the action outputs stop signal#
        #while True:
        ############################################
            # print('s', s)
            # print('masked_actions', masked_actions)
            s = s.reshape(1,-1)
            prob = model.actor(torch.from_numpy(s).float()) # s => tensor #output = prob for actions


            # a = Categorical(prob).sample().numpy() #substitute
            # print('a!!! ====>' , a)

            #Choose the action with highest prob and not in masked action
            #Becky#########################################################
            prob = prob.cpu().detach().numpy().reshape(-1,)
            #Check if the action is valid
            action_Invalid = True
            largest_num = -1
            while action_Invalid:
                a = prob.argsort()[largest_num:][0]
                action_Invalid = True if masked_actions[a] == 0 else False
                a = [a]
                largest_num -= 1
            # print('masked_actions', masked_actions)
            # print('a', a)
            #Check the action is a stop sign or not a = [0] means stop
            # if a == 0:
            #     break
            # done = np.array([0])
            ###############################################################
            with HiddenPrints():
                s_prime, r, done, masked_actions = envs.step(a)
            # print('done =>', done)
            s_prime = s_prime.reshape(1,-1)
            # print('s_prime', s_prime)
            masked_actions = masked_actions[0]

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r/100.0) #discount of actions, hyperparameter
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

            num += 1

        s_final = torch.from_numpy(s_prime).float() #numpy => tensor
        v_final = model.critic(s_final).detach().clone().numpy() #V(s') numpy  i.e. [[0.09471023]]
        td_target = compute_target(v_final, r_lst, mask_lst, gamma=0.98) #hyperparameter gamma

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(s_lst).float().reshape(-1, config.state_num)  # total states sequence under a sequence of actions =>tensor [[]]
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1) #a sequence of actions =>tensor [[]]
        advantage = td_target_vec - model.critic(s_vec).reshape(-1) #  advantage function to update

        probs_all_state = model.actor(s_vec, softmax_dim=1)
        probs_actions = probs_all_state.gather(1, a_vec).reshape(-1) #tensor i.e. tensor([...,...,...])
        loss = -(torch.log(probs_actions) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.critic(s_vec).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)
    # print('s_lst', s_lst)
    envs.close()