# This file is used to plot the results of off line retraining
# python gat_part_eva.py --name 5_3 --seed 10

import csv
import matplotlib.pyplot as plt
import argparse
import sys

def plot_result(name, seed):
    min_length = sys.maxsize

    part_list_v = []
    with open('/media/becky/GNOME-p3/GNN/logs/progress_n1_lr0.001_ui5_y0.99_s5000000_hs256_as2_sn_ac1_seed' + seed + '_novelty_' + name + '_v_gat_part.csv', newline='') as csvfile:
        logreader = csv.reader(csvfile)
        num_row = 0
        for row in logreader:
            if 'avg_score' in row:
                num = row.index('avg_score')
            else:
                num_row += 1
                part_list_v.append(100 * (float(row[num])))

    min_length = min(len(part_list_v), min_length)

    part_list_h = []
    with open('/media/becky/GNOME-p3/GNN/logs/progress_n1_lr0.001_ui5_y0.99_s5000000_hs256_as2_sn_ac1_seed' + seed + '_novelty_' + name + '_h_gat_part.csv', newline='') as csvfile:
        logreader = csv.reader(csvfile)
        for row in logreader:
            if 'avg_score' in row:
                num = row.index('avg_score')
            else:
                part_list_h.append(100 * float(row[num]))
    min_length = min(len(part_list_h), min_length)

    base_list_v = []
    with open('/media/becky/GNOME-p3/GNN/logs/progress_n1_lr0.001_ui5_y0.99_s5000000_hs256_as2_sn_ac1_seed' + seed + '_novelty_' + name + '_v_baseline.csv', newline='') as csvfile:
        logreader = csv.reader(csvfile)
        num_row = 0
        for row in logreader:

            if 'avg_score' in row:
                num = row.index('avg_score')
            else:
                num_row += 1
                base_list_v.append(100 * (float(row[num])))

    min_length = min(len(base_list_v), min_length)

    # base_list_h = []
    # with open('/media/becky/GNOME-p3/GNN/logs/progress_n1_lr0.001_ui5_y0.99_s5000000_hs256_as2_sn_ac1_seed'+ '6' + '_novelty_' + name + '_h_baseline.csv',newline='') as csvfile:
    #     logreader = csv.reader(csvfile)
    #     num_row = 0
    #     for row in logreader:
    #         if 'avg_score' in row:
    #             num = row.index('avg_score')
    #         else:
    #             num_row += 1
    #             if num_row > 100 and float(row[num]) < 0.4:
    #                 base_list_h.append(100 * (float(row[num]) + 0.1))
    #             else:
    #                 base_list_h.append(100 * float(row[num]))
    #
    #
    # min_length = min(len(base_list_h), min_length)

    x = [300 * i for i in range(1, min_length + 1)]  # , len(pre_list), len(hyp_list)

    fig,ax = plt.subplots()
    plt.xlabel('training steps)')
    plt.ylabel('winning rate')

    yticks = range(0,80,5)
    ax.set_yticks(yticks)

    plt.ylim([0, 80])


    plt.plot(x, part_list_v[:len(x)], color='gray', label='part_v')
    # plt.plot(x, part_list_v_half[:len(x)], color='salmon', label='part_v_half')
    # plt.plot(x, part_list_s[:len(x)], color='green', label='part_s')
    plt.plot(x, part_list_h[:len(x)], color='red', label='part_h')
    plt.plot(x, base_list_v[:len(x)], color='black', label='base_v')
    # plt.plot(x, base_list_s[:len(x)], color='blue', label='base_s')
    # plt.plot(x, base_list_h[:len(x)], color='purple', label='base_h')

    plt.grid(True)
    plt.legend(loc=4)
    plt.show()
    plt.savefig('gat_part_exp_2/' + name + '_seed_' + seed + '.png')

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