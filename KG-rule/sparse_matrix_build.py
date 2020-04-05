import numpy as np
from scipy.sparse import csr_matrix

# class Action_Relation():
#     def __init__(self):
#         self.sparse_matrix = [csr_matrix((39, 39), dtype=np.int8)]
sparse_matrix = [csr_matrix((39, 39), dtype=np.int8) for i in range(20)]
print(sparse_matrix)