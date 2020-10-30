import pickle
import numpy as np
# with open('/media/becky/GNOME-p3/KG_rule/matrix_rule/kg_matrix_0_0_kg.npy', 'rb') as f:
#     nodes = pickle.load(f)
#     print(nodes[12])
#     print(sorted(list(nodes.values())))
#     print(len(nodes))
matrix = np.load('/media/becky/GNOME-p3/Evaluation_2/monopoly_simulator_2/A2C_agent_2/matrix/matrix.npy')
print(matrix.shape)
rel_list = ['is priced at', 'is price-1-house at', 'is rented-0-house at', 'is rented-1-house at', \
                                 'is rented-2-house at', 'is rented-3-house at', 'is rented-4-house at', 'is rented-1-hotel', 'color']
range_list = [(40,47), (47,52), (52,59), (59,64), (64,72), (72,80), (80,86), (86,matrix.shape[0] + 1), (0, 40)]
for i in range(6,7):
    matrix_node = matrix[i]
    for j in range(9):
        rel = matrix_node[j * matrix.shape[0] : (j + 1) * matrix.shape[0]]
        print(rel_list[j])
        a,b = range_list[j]
        print(rel[a:b])


# load_path = '/media/becky/GNOME-p3/KG_rule/matrix_rule/entity_id_30_1_tryenv.json'
# # load_path = '/media/becky/GNOME-p3/KG_rule/matrix_rule/entity_id_0_0_baseline_ran.json'
# with open(load_path, 'rb') as f:
#     load_dict = pickle.load(f)
#     print(load_dict)
