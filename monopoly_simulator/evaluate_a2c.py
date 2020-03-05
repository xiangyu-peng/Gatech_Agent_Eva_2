from vanilla_A2C import *
from configparser import ConfigParser
import graphviz
from torchviz import make_dot


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
    update_interval = 5
    gamma = 0.98
    max_train_steps = 60000
    PRINT_INTERVAL = update_interval * 10
    config = Config()
    config.hidden_state = 256
    config.action_space = 80
    config.state_num = 56
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

    model_path = '/media/becky/GNOME-p3/monopoly_simulator/weights/push_buy.pkl'
    model = torch.load(model_path)
    test(1, model, device)