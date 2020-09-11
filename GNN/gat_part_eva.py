# This file is used to plot the results of off line retraining
# python gat_part_eva.py --name 5_3 --seed 10

import csv
import matplotlib.pyplot as plt
import argparse
import sys

def plot_result(name, seed):
    min_length = sys.maxsize
    part_list = []
    with open('/media/becky/GNOME-p3/GNN/logs/progress_n2_lr0.001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed' + seed + '_novelty_' + name + '_gat_part.csv', newline='') as csvfile:
        logreader = csv.reader(csvfile)
        for row in logreader:
            if 'avg_score' in row:
                num = row.index('avg_score')
            else:
                part_list.append(100 * float(row[num]))
    min_length = min(len(part_list), min_length)


    base_list = []
    with open('/media/becky/GNOME-p3/GNN/logs/progress_n2_lr0.001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed' + seed + '_novelty_' + name + '_baseline.csv', newline='') as csvfile:
        logreader = csv.reader(csvfile)
        for row in logreader:
            if 'avg_score' in row:
                num = row.index('avg_score')
            else:
                base_list.append(100 * float(row[num]))
    min_length = min(len(base_list), min_length)


    x = [1000 * i for i in range(1, min_length + 1)]  # , len(pre_list), len(hyp_list)

    fig,ax = plt.subplots()
    plt.xlabel('training steps)')
    plt.ylabel('winning rate')

    yticks = range(20,70,5)
    ax.set_yticks(yticks)

    plt.ylim([20, 65])


    plt.plot(x, part_list[:len(x)], color='green', label='part')

    plt.plot(x, base_list[:len(x)], color='black', label='base')


    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('gat_part_result/' + name + '_seed_' + seed + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='5_3', type=str,
                        required=True,
                        help="novelty name, i.g. 5_3")
    parser.add_argument('--seed',
                        default='0', type=str,
                        required=True,
                        help="seed for training")
    args = parser.parse_args()
    plot_result(args.name, args.seed)