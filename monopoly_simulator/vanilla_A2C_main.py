from vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot
from gameplay_simple_tf import *

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

if __name__ == '__main__':
    config_file = '/media/becky/Novelty-Generation-Space-A2C/Vanilla-A2C/config.ini'
    config_data = ConfigParser()
    config_data.read(config_file)
    # print('config_data.items', config_data.sections())
    # Hyperparameters
    n_train_processes = 1
    learning_rate = 0.0002
    update_interval = 1
    gamma = 0.98
    max_train_steps = 60000
    PRINT_INTERVAL = 30
    config = Config()
    config.hidden_state = 256
    config.action_space = 80
    config.state_num = 56
    actor_loss_coefficient = 1
    save_dir = '/media/becky/GNOME-p3/monopoly_simulator'
    device = torch.device('cuda:0')
    save_name = '/push_buy'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # torch.cuda.set_device(1)
    # device = torch.device("cuda", 1)

    #######################################
    with HiddenPrints():
        envs = ParallelEnv(n_train_processes)
    model = ActorCritic(config) #A2C model
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    with HiddenPrints():
        s, masked_actions = envs.reset()
        # s = torch.tensor(s, device=device).float()
        # masked_actions = torch.tensor(masked_actions, device=device)


    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list() #state list; action list, reward list, masked action list？？？


        for _ in range(update_interval): #substitute
        ########Becky###############################
        ##loop until the action outputs stop signal#
        #while True:
        ############################################

            s = s.reshape(1, -1)
            prob = model.actor(torch.tensor(s, device=device).float())  # s => tensor #output = prob for actions
            # prob = model.actor(torch.from_numpy(s,device=device).float()) # s => tensor #output = prob for actions

            #use the action from the distribution if it is in masked
            action_Invalid = True
            while action_Invalid:
                a = Categorical(prob).sample().cpu().numpy() #substitute
                action_Invalid = True if masked_actions[a[0]] == 0 else False
            # print('masked_actions', masked_actions)
            # print('=============================')
            #while opposite happens. Step won't change env, step_nochange changes the env
            s_prime, r, done, _ = envs.step_nochange(a)
            # print('s_prime', s_prime)
            # print('done', done)
            # print('=============================')



            #Choose the action with highest prob and not in masked action
            #Becky#########################################################
            # prob = prob.cpu().detach().numpy().reshape(-1,)
            # #Check if the action is valid
            # action_Invalid = True
            # largest_num = -1
            # while action_Invalid:
            #     a = prob.argsort()[largest_num:][0]
            #     action_Invalid = True if masked_actions[a] == 0 else False
            #     a = [a]
            #     largest_num -= 1
            # print('a =====>', a)
            #Check the action is a stop sign or not a = [0] means stop
            # if a == 0:
            #     break
            # done = np.array([0])
            ###############################################################
            # with HiddenPrints():
            #     s_prime, r, done, masked_actions = envs.step(a)



            # print('reward', r)
            # if done:
            #     print(s_prime)
                # print('s_prime, r, done, masked_actions', s_prime, r, done, masked_actions)




            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r) # r/100 discount of actions, hyperparameter
            if done:
                mask_lst.append(0)
            else:
                mask_lst.append(1)

            a_tf = tf(s[0], masked_actions)
            # print(a_tf)
            # print('****************************')
            s_prime, _, done, masked_actions = envs.step(a_tf)
            # print('s_prime', s_prime)
            # print('done', done)
            # print('****************************')
            masked_actions = masked_actions[0]
            s_prime = s_prime.reshape(1, -1)
            s = s_prime
            step_idx += 1

            if done:
                # print('s_prime, r, done, masked_actions', s_prime, r, done, masked_actions)
                with HiddenPrints():
                    s, masked_actions = envs.reset()
                break


        s_final = torch.tensor(s_prime, device = device).float() #numpy => tensor
        v_final = model.critic(s_final).detach().clone().cpu().numpy() #V(s') numpy  i.e. [[0.09471023]]
        td_target = compute_target(v_final, r_lst, mask_lst, gamma=0.98) #hyperparameter gamma

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(s_lst, device = device).float().reshape(-1, config.state_num)  # total states sequence under a sequence of actions =>tensor [[]]
        a_vec = torch.tensor(a_lst, device = device).reshape(-1).unsqueeze(1) #a sequence of actions =>tensor [[]]
        advantage = td_target_vec - model.critic(s_vec).cpu().reshape(-1) #  advantage function to update
        advantage = advantage.to(device)


        probs_all_state = model.actor(s_vec, softmax_dim=1)
        probs_actions = probs_all_state.gather(1, a_vec).reshape(-1) #tensor i.e. tensor([...,...,...])
        loss = -(torch.log(probs_actions) * advantage.detach()).mean() * actor_loss_coefficient +\
            F.smooth_l1_loss(model.critic(s_vec).reshape(-1), td_target_vec.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        graphviz.Source(make_dot(loss, params=dict(model.named_parameters()))).render('full_net')
        # print('weight after test = > ', model.fc_actor.weight)
        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model,device, num_test=10)
            #save weights of A2C
            save_path = '/media/becky/GNOME-p3/monopoly_simulator/weights'
            save_name = save_path + '/push_buy.pkl'

            torch.save(model, save_name)

            # print('weight after test = > ', model.fc_actor.weight)
    # print('s_lst', s_lst)
    envs.close()