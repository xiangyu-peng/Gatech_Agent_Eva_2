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
    def __init__(self, input_dim, h_dim, drop_prob, support, num_bases, device, feature_dict=None, featureless=False, bias=False):
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
        self.device = device
        self.features = None
        if not featureless:
            self.features = self.get_feature(feature_dict)

        # these will be defined during build()
        if self.num_bases > 0:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.num_bases, self.h_dim, dtype=torch.float32, device=device))
            self.W_comp = nn.Parameter(torch.empty(self.support, self.num_bases, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.W_comp)
        else:
            if not featureless:
                self.W = nn.Parameter(
                    torch.empty(self.features.shape[1] * self.support, self.h_dim, dtype=torch.float32, device=device))
            else:
                self.W = nn.Parameter(
                    torch.empty(self.input_dim * self.support, self.h_dim, dtype=torch.float32, device=device))
        # initialize the weight
        nn.init.xavier_uniform_(self.W)
        # initialize the bias if necessary
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.h_dim, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.b)

    def assign_bin(self, np_array, row, class_total, feature_total, class_id, feature_id):
        """
        :param np_array:
        :param row:
        :param class_total: int length
        :param feature_total: int length
        :param class_id: int
        :param feature_id: int
        :return:
        """
        class_id = bin(class_id).replace('0b', '')
        while len(class_id) < class_total:
            class_id = '0' + class_id
        feature_id = bin(feature_id).replace('0b', '')

        while len(feature_id) < feature_total:
            feature_id = '0' + feature_id

        for id in range(class_total):
            np_array[row][id] = int(class_id[id])
        for id_f in range(feature_total):
            np_array[row][id_f + id + 1] = int(feature_id[id_f])
        np_array = torch.tensor(np_array, device=self.device)
        return np_array

    def get_feature(self, feature_dict):
        features = np.zeros((self.input_dim,9))
        property_id = 0
        for id in range(len(feature_dict)):
            if 'player' == feature_dict[id]:
                # class_id = 1; feature_id = 1
                class_id = 1
                feature_id = 1

            elif 'player_' in feature_dict[id]:
                class_id = 2
                feature_id = int(feature_dict[id].replace('player_', ''))

            elif 'House_' in feature_dict[id]:
                class_id = 4
                feature_id = int(feature_dict[id].replace('House_', '')) + 1

            elif 'Cash_' in feature_dict[id]:
                class_id = 5
                feature_id = int(feature_dict[id].replace('Cash_', '')) + 1

            else:
                property_id += 1
                class_id = 3
                feature_id = property_id

            features = self.assign_bin(features,
                                       id,
                                       class_total=3,
                                       feature_total=6,
                                       class_id=class_id,
                                       feature_id=feature_id)
        return features

    def forward(self, adj, feature=None):
        features = feature if self.featureless else self.features
        A = adj  # graph relationship
        A_hat = torch.tensor(adj + np.array([np.eye(len(adj[0])) for i in range(len(adj))]), device=self.device)
        # # normalization
        # D = np.array(np.sum(A, axis=2))
        # D = np.array([np.diag(i) for i in D]).astype(np.float32)
        # print('D', D.shape)
        # print(D ** -1)
        # convolve
        supports = list()
        # print(A_hat.shape, features.shape)
        if self.featureless:
            for i in range(A_hat.shape[0]):
                supports.append(torch.mm(A_hat[i], features[i]))
        else:
            for i in range(A_hat.shape[0]):
                supports.append(torch.mm(A_hat[i], features))
        # supports = torch.cat(supports, dim=1)
        # print(supports)

        if self.num_bases > 0:
            V = torch.matmul(self.W_comp,
                             self.W.reshape(self.num_bases, self.input_dim, self.h_dim).permute(1, 0, 2))
            V = torch.reshape(V, (self.support * self.input_dim, self.h_dim))
            output = torch.spmm(supports, V)
        else:
            output = []
            for i in range(len(supports)):
                output.append(torch.mm(supports[i].float(), self.W))
            # output = torch.tensor(output)
            # print(output.shape)
        # if featureless add dropout to output, by elementwise matmultiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        # num_nodes = supports.shape[0] # Dan: or features.shape[1]

        # if self.featureless:
        #     tmp = torch.ones(num_nodes)
        #     tmp_do = self.dropout(tmp)
        #     output = (output.transpose(1, 0) * tmp_do).transpose(1, 0)

        if self.bias:
            for i in range(len(output)):
                output[i] += self.b
                output[i] = self.activation(output[i])
        return output

class RGCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_prob, support, num_bases, device, output_size, feature_dict):
        super(RGCNetwork, self).__init__()
        self.device = device
        self.gcl_1 = RGCLayer(input_dim, hidden_dim, drop_prob, support, num_bases, device, feature_dict,
                              featureless=False, bias=False)
        self.gcl_2 = RGCLayer(hidden_dim, output_size, drop_prob, support, num_bases, device,
                              featureless=True, bias=False)

        self.dropout = nn.Dropout(drop_prob)
        # self.fc1 = nn.Linear(hidden_dim, output_size).cuda()  # what is 3?

    def forward(self, adj):
        output = self.gcl_1(adj, feature=None)
        output = [output[i].cpu().detach().numpy() for i in range(len(output))]
        output = torch.tensor(output, dtype=torch.float64, device=self.device)

        output = self.gcl_2(adj, feature=output)
        output = [output[i].cpu().detach().numpy() for i in range(len(output))]
        output = torch.tensor(output, device=self.device)

        # output = self.dropout(output)
        # ret = self.fc1(output)
        return output


if __name__ == '__main__':
    device = torch.device('cuda:0')
    # import json
    # with open('KG-rule/json_kg.json', 'r') as f:
    #     kg_data = json.load(f)

    import numpy as np
    outfile = '/media/becky/GNOME-p3/KG_rule/matrix_rule/kg_matrix_10_1_state.npy'
    npzfile = np.load(outfile)
    for i in range(npzfile.shape[1] // npzfile.shape[0] - 1):
         npzfile =  npzfile[:,  npzfile.shape[0] * i: npzfile.shape[0] * (i + 1)] +  npzfile[:,
                                                                                     npzfile.shape[0] * (i + 1):
                                                                                     npzfile.shape[0] * (i + 2)]

    input_dim = npzfile.shape[0]
    hidden_dim = 20
    drop_prob = 0.1
    support = 1
    num_bases = -1
    output_size = 10
    nnk = RGCNetwork(input_dim, hidden_dim, drop_prob, support, num_bases, device, output_size)
    oo = nnk.forward(npzfile)
    print(oo)
    print(oo.shape)


