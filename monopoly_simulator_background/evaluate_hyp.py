import csv
hyp_list = []
name = '1_5'
with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_hyp.csv', newline='') as csvfile:
    logreader = csv.reader(csvfile)
    for row in logreader:
        if 'avg_winning' in row:
            num = row.index('avg_winning')
        else:
            hyp_list.append(100 * float(row[num]))

# hyp_list_10 = []
# with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_hyp_10.csv', newline='') as csvfile:
#     logreader = csv.reader(csvfile)
#     for row in logreader:
#         if 'avg_winning' in row:
#             num = row.index('avg_winning')
#         else:
#             hyp_list_10.append(100 * float(row[num]))


pre_list = []
with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_pre.csv', newline='') as csvfile:
    logreader = csv.reader(csvfile)
    for row in logreader:
        if 'avg_winning' in row:
            num = row.index('avg_winning')
        else:
            pre_list.append(100 * float(row[num]))

ran_list = []
with open('/media/becky/GNOME-p3/monopoly_simulator_background/logs/progress_n2_lr0.0001_ui5_y0.99_s500000_hs256_as2_sn_ac1_seed8_novelty_' + name + '_ran.csv', newline='') as csvfile:
    logreader = csv.reader(csvfile)
    for row in logreader:
        if 'avg_winning' in row:
            num = row.index('avg_winning')
        else:
            ran_list.append(100 * float(row[num]))

x = [1000 * i for i in range(1, min(len(ran_list), len(pre_list), len(hyp_list)) + 1)]  # , len(hyp_list_10)


import matplotlib.pyplot as plt

fig,ax = plt.subplots()
plt.xlabel('training steps)')
plt.ylabel('winning rate')

yticks = range(20,60,5)
ax.set_yticks(yticks)

plt.ylim([20, 55])
# ax.set_xlim([58,42])

plt.plot(x, hyp_list[:len(x)], color='green', label='hyp')
# plt.plot(x, hyp_list_10[:len(x)], color='black', label='hyp_10')
plt.plot(x, pre_list[:len(x)], color='blue', label='pre')
plt.plot(x, ran_list[:len(x)], color='red', label='ran')

plt.grid(True)
plt.legend()
plt.show()
plt.savefig('hyp_result/' + name + '.png')
