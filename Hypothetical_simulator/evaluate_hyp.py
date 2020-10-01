# This file is used to plot the results of off line retraining
# python evaluate_hyp.py --interval 10 --novelty 4,19_5,3 --type pre --seed 10

import csv
import matplotlib.pyplot as plt
import argparse
import sys

def plot_result(name, type_list, interval, seed):
    if 'all' in type_list:
        type_list = ['hyp_gat_pre', 'gat_pre', 'baseline_pre', 'hyp_pre', 'baseline_ran', 'gat_ran', 'hyp_gat_ran', 'hyp_ran']
    if 'ran' in type_list:
        type_list = ['baseline_ran', 'gat_ran', 'hyp_gat_ran','hyp_ran']
    if 'pre' in type_list:
        type_list = ['hyp_gat_pre', 'gat_pre', 'baseline_pre', 'hyp_pre']
    min_length = sys.maxsize

    if 'hyp_gat_pre' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        hyp_gat_pre_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_hyp_gat_pre.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_gat_pre_list.append(100 * float(row[num]))
        min_length = min(len(hyp_gat_pre_list), min_length)

    if 'gat_pre' in type_list:
        """
        Not use gat
        retrain from pre
        w/ hyp sim
        """
        gat_pre_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_gat_pre.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    gat_pre_list.append(100 * float(row[num]))
        min_length = min(len(gat_pre_list), min_length)

    if 'baseline_pre' in type_list:
        """
        Not use gat
        retrain from scratch
        w/o hyp sim
        """
        baseline_pre_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_baseline_pre.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    baseline_pre_list.append(100 * float(row[num]))
        min_length = min(len(baseline_pre_list), min_length)

    if 'hyp_pre' in type_list:
        """
        Use gat
        retrain from pre
        w/ hyp sim
        """
        hyp_pre_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_hyp_pre.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_pre_list.append(100 * float(row[num]))
        min_length = min(len(hyp_pre_list), min_length)

    if 'baseline_ran' in type_list:
        """
        Use gat
        retrain from scratch
        w/o hyp sim
        """
        baseline_ran_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_baseline_ran.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    baseline_ran_list.append(100 * float(row[num]))
        min_length = min(len(baseline_ran_list), min_length)

    if 'gat_ran' in type_list:
        """
        Use gat
        retrain from scratch
        w/o hyp sim
        """
        gat_ran_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_gat_ran.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    gat_ran_list.append(100 * float(row[num]))
        min_length = min(len(gat_ran_list), min_length)

    if 'hyp_gat_ran' in type_list:
        """
        Use gat
        retrain from scratch
        w/o hyp sim
        """
        hyp_gat_ran_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_hyp_gat_ran.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_gat_ran_list.append(100 * float(row[num]))
        min_length = min(len(hyp_gat_ran_list), min_length)

    if 'hyp_ran' in type_list:
        """
        Use gat
        retrain from scratch
        w/o hyp sim
        """
        hyp_ran_list = []
        with open('/media/becky/GNOME-p3/Hypothetical_simulator/log_test/progress_seed_' + seed + '_nov_' + name + '_hyp_ran.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile)
            for row in logreader:
                if 'avg_winning' in row:
                    num = row.index('avg_winning')
                else:
                    hyp_ran_list.append(100 * float(row[num]))
        min_length = min(len(hyp_ran_list), min_length)

    x = [interval  * i for i in range(1, min_length + 1)]  # , len(pre_list), len(hyp_list)

    fig,ax = plt.subplots()
    plt.xlabel('training steps)')
    plt.ylabel('winning rate')

    yticks = range(0,70,5)
    # xticks = range(0, max(x), 1)
    ax.set_yticks(yticks)
    # ax.set_yticks(xticks)

    plt.ylim([0, 65])

    if 'hyp_gat_pre' in type_list:
        plt.plot(x, hyp_gat_pre_list[:len(x)], color='red', label='hyp_gat_pre')
    if 'gat_pre' in type_list:
        plt.plot(x, gat_pre_list[:len(x)], color='blue', label='gat_pre')
    if 'baseline_pre' in type_list:
        plt.plot(x, baseline_pre_list[:len(x)], color='black', label='baseline_pre')
    if 'hyp_pre' in type_list:
        plt.plot(x, hyp_pre_list[:len(x)], color='green', label='hyp_pre')
    if 'baseline_ran' in type_list:
        plt.plot(x, baseline_ran_list[:len(x)], color='darkslategray', label='baseline_ran')
    if 'gat_ran' in type_list:
        plt.plot(x, gat_ran_list[:len(x)], color='cadetblue', label='gat_ran')
    if 'hyp_gat_ran' in type_list:
        plt.plot(x, hyp_gat_ran_list[:len(x)], color='orange', label='hyp_gat_ran')
    if 'hyp_ran' in type_list:
        plt.plot(x, hyp_ran_list[:len(x)], color='palegreen', label='hyp_ran')


    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('hyp_result/' + name + '_' + seed + '.png')

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
    parser.add_argument('--plot_interval', type=int,
                        default=1, required=False,
                        help="add multiple plot type, use comma to split, i.e. hyp,ran")
    parser.add_argument('--seed',
                        default='10', type=str,
                        required=False,
                        help="retraining step, i.g. 10")
    args = parser.parse_args()
    novelty_name = str(list(map(lambda x : int(x), args.novelty.split('_')[0].split(',')))) + '_' + str(list(map(lambda x : int(x),args.novelty.split('_')[1].split(','))))
    name = novelty_name + '_rt_' + args.interval
    plot_result(name, args.type.split(','), args.plot_interval, args.seed)