import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
from scipy import sparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# from the R-GCN/rgcn_pytorch
class RGCLayer(nn.Module):
    def __init__(self, input_dim, h_dim, drop_prob, support, num_bases, featureless=True, bias=False):
        super(RGCLayer, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim  # number of features per node
        self.dropout = nn.Dropout(drop_prob)
        self.support = support  # filter support / number of weights
        assert support >= 1, 'support must be no smaller than 1'
        self.num_bases = num_bases
        self.featureless = featureless # use/ignore input features
        self.bias = bias
        self.activation = nn.ReLU() # default

        # these will be defined during build()
        if self.num_bases > 0:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.num_bases, self.h_dim, dtype=torch.float32, device=device))
            self.W_comp = nn.Parameter(torch.empty(self.support, self.num_bases, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.W_comp)
        else:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.support, self.h_dim, dtype=torch.float32, device=device))
        # initialize the weight
        nn.init.xavier_uniform_(self.W)
        # initialize the bias if necessary
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.h_dim, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.b)

    def forward(self, inputs):
        features = torch.tensor(inputs[0], dtype=torch.float32, device=device)
        A = inputs[1:]  # list of basis functions, original inputs is: [X] + A
        A = [torch.sparse.FloatTensor(torch.LongTensor(a.nonzero())
                                      , torch.FloatTensor(sparse.find(a)[-1])
                                      , torch.Size(a.shape)).to(device)
             if len(sparse.find(a)[-1]) > 0 else torch.sparse.FloatTensor(a.shape[0], a.shape[1])
             for a in A] #all sparse matrix
        # for a in A:
        #     print('A===>', a)
        #     break

        # convolve
        if not self.featureless:
            supports = list()
            for i in range(self.support):
                supports.append(torch.spmm(A[i], features))
            supports = torch.cat(supports, dim=1)
        else:
            values = torch.cat([i._values() for i in A], dim=-1) # all the values in sparse matrix A into one tensor
            indices = torch.cat([torch.cat([j._indices()[0].reshape(1, -1),
                                            (j._indices()[1] + (i * self.input_dim)).reshape(1, -1)])
                                 for i, j in enumerate(A)], dim=-1) #indices for values, 2*n
            #nodes * (nodes*actions) big matrix
            supports = torch.sparse.FloatTensor(indices, values, torch.Size([A[0].shape[0],
                                                                          len(A) * self.input_dim]))
        if self.num_bases > 0:
            V = torch.matmul(self.W_comp,
                             self.W.reshape(self.num_bases, self.input_dim, self.h_dim).permute(1, 0, 2))
            V = torch.reshape(V, (self.support * self.input_dim, self.h_dim))
            output = torch.spmm(supports, V)
        else:
            output = torch.spmm(supports, self.W)
        # if featureless add dropout to output, by elementwise matmultiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        num_nodes = supports.shape[0] # Dan: or features.shape[1]
        if self.featureless:
            tmp = torch.ones(num_nodes)
            tmp_do = self.dropout(tmp)
            output = (output.transpose(1, 0) * tmp_do).transpose(1, 0)

        if self.bias:
            output += self.b
        return self.activation(output)


class RGCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_prob, support, num_bases):
        super(RGCNetwork, self).__init__()
        self.gcl_1 = RGCLayer(input_dim, hidden_dim, drop_prob, support, num_bases,
                              featureless = True, bias = False)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs):
        output = self.gcl_1(inputs)
        output = self.dropout(output)
        return output


if __name__ == '__main__':
    # import json
    # with open('KG-rule/json_kg.json', 'r') as f:
    #     kg_data = json.load(f)

    import numpy as np
    outfile = 'KG-rule/matrix_rule/is10-stepawayfrom.npz'
    npzfile = np.load(outfile)
    print(npzfile.files)
    print(npzfile['indices'])
    print(npzfile['indptr'])
    print(npzfile['format'])
    print(npzfile['shape'])
    print(npzfile['data'])
    import scipy.sparse
    sparse_matrix = scipy.sparse.load_npz(outfile)
    print(sparse_matrix)
    # print(sparse_matrix.todense())

    from scipy.sparse import csr_matrix
    kg_matrix = [csr_matrix((39, 39), dtype=np.int8) for i in range(39)]

    input_dim = kg_matrix[0].shape[0]
    hidden_dim = 20
    drop_prob = 0.3
    support = len(kg_matrix)
    num_bases = 3

    RGCNetwork(input_dim, hidden_dim, drop_prob, support, num_bases)




