# This file is used to plot the results of off line retraining
# python evaluate_hyp.py --novelty 5,20,5,20_4,1,4,1 --interval 10 --type ran,gat

import csv
import matplotlib.pyplot as plt
import argparse
import sys

def plot_result(name, type_list):
    min_length = sys.maxsize

    if 'hyp' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        hyp_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_10_nov_' + name + '_hyp.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_list.append(100 * float(row[num]))
        min_length = min(len(hyp_list), min_length)

    if 'baseline' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        pre_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_10_nov_' + name + '_baseline.csv', newline='') as csvfile:
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
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_10_nov_' + name + '_baseline.csv', newline='') as csvfile:
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
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_10_nov_' + name + '_kg_hyp_stop_5k.csv', newline='') as csvfile:
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
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_10_nov_' + name + '_gat.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    gat_list.append(100 * float(row[num]))
        min_length = min(len(gat_list), min_length)

    x = [50 * i for i in range(1, min_length + 1)]  # , len(pre_list), len(hyp_list)

    fig,ax = plt.subplots()
    plt.xlabel('training steps)')
    plt.ylabel('winning rate')

    yticks = range(0,70,5)
    # xticks = range(0, max(x), 1)
    ax.set_yticks(yticks)
    # ax.set_yticks(xticks)

    plt.ylim([0, 65])

    if 'hyp' in type_list:
        plt.plot(x, hyp_list[:len(x)], color='green', label='hyp')
    if 'baseline' in type_list:
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
    parser.add_argument('--novelty',
                        default='5_3', type=str,
                        required=True,
                        help="novelty name, i.g. 5,18,5,18_3,1,3,1")
    parser.add_argument('--interval',
                        default='10', type=str,
                        required=True,
                        help="retraining step, i.g. 10")
    parser.add_argument('--type', type=str,
                        default=None, required=True,
                        help="add multiple plot type, use comma to split, i.e. hyp,ran")
    args = parser.parse_args()
    novelty_name = str(list(map(lambda x : int(x), args.novelty.split('_')[0].split(',')))) + '_' + str(list(map(lambda x : int(x),args.novelty.split('_')[1].split(','))))
    name = novelty_name + '_rt_' + args.interval
    plot_result(name, args.type.split(','))