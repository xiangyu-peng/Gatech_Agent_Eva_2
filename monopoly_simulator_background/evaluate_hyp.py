# This file is used to plot the results of off line retraining
# python evaluate_hyp.py --name 9_4 --type gat,ran

import csv
import matplotlib.pyplot as plt
import argparse
import sys

def plot_result(name, type_list):
    min_length = sys.maxint
    if 'hyp' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        hyp_list = []
        with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_hyp.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_list.append(100 * float(row[num]))
        min_length = min(len(hyp_list), min_length)

    if 'pre' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        pre_list = []
        with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_pre.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    pre_list.append(100 * float(row[num]))
        min_length = min(len(pre_list), min_length)

    if 'ran' in type_list:
        """
        Not use gat
        retrain from scratch
        w/o hyp sim
        """
        ran_list = []
        with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_ran.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    ran_list.append(100 * float(row[num]))
        min_length = min(len(ran_list), min_length)

    if 'gat_hyp' in type_list:
        """
        Use gat
        retrain from pre
        w/ hyp sim
        """
        gat_hyp_list = []
        with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed10_novelty_' + name + '_kg_hyp_stop_5k.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    gat_hyp_list.append(100 * float(row[num]))
        min_length = min(len(gat_hyp_list), min_length)

    if 'gat' in type_list:
        """
        Use gat
        retrain from scratch
        w/o hyp sim
        """
        gat_list = []
        with open('/media/becky/GNOME-p3/GNN/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed10_novelty_' + name + '_gat_ran_kg.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    gat_list.append(100 * float(row[num]))
        min_length = min(len(gat_list), min_length)

    x = [1000 * i for i in range(1, min_length + 1)]  # , len(pre_list), len(hyp_list)

    fig,ax = plt.subplots()
    plt.xlabel('training steps)')
    plt.ylabel('winning rate')

    yticks = range(20,70,5)
    ax.set_yticks(yticks)

    plt.ylim([20, 65])

    if 'hyp' in type_list:
        plt.plot(x, hyp_list[:len(x)], color='green', label='hyp')
    if 'pre' in type_list:
        plt.plot(x, pre_list[:len(x)], color='blue', label='pre')
    if 'ran' in type_list:
        plt.plot(x, ran_list[:len(x)], color='black', label='ran')
    if 'gat_hyp' in type_list:
        plt.plot(x, gat_hyp_list[:len(x)], color='red', label='gat_hyp')
    if 'gat' in type_list:
        plt.plot(x, gat_list[:len(x)], color='salmon', label='gat')

    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('hyp_result/' + name + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='5_3', type=str,
                        required=True,
                        help="novelty name, i.g. 5_3")
    parser.add_argument('--type', type=str,
                        default=None, required=True,
                        help="add multiple plot type, use comma to split, i.e. hyp,ran")
    args = parser.parse_args()
    plot_result(args.name, args.type.split(','))