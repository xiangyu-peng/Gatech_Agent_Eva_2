# This class is designed for generating the adj which is related with the observable world
import numpy as np
class Adj_Gen(object):
    def __init__(self, adj=None, patience=4):
        self.adj = adj  # a square/ n by n np matrix
        self.patience = patience  # the window of position; patience=1 means only see this node's info

    def update_adj(self, adj):
        if self.adj is not None:
            comparison = adj == self.adj  # assume the size of adj stay the same
            if comparison.all():
                return
        self.adj = adj

    def output_part_adj(self, adj, position):
        self.update_adj(adj)
        output_adj = self.adj.copy()
        for i in range(output_adj.shape[0]):
            if i < position or i >= position + self.patience:
                output_adj[i][output_adj[i] > 0] = 0  # mask all the nodes outside patience window

        # make the matrix as symetric
        for i in range(output_adj.shape[0]):
            for j in range(output_adj.shape[0]):
                if output_adj[i][j] != 0:
                    output_adj[j][i] = output_adj[i][j]

        return output_adj

    def get_pos(self, state):
        n_state = len(state)
        pos_list = []
        for n in range(n_state):
            try:
                pos_list.append(state[n][80:120].cpu().numpy().tolist().index(1))
            except:
                pos_list.append(0)
        return pos_list

if __name__ == '__main__':  # unit test
    adj = np.array([[0,0,1],[1,1,0],[1,1,1]])
    adj_gen = Adj_Gen(adj, patience=1)
    print(adj_gen.output_part_adj(adj, 0))