#####Add Path#####
import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import spacy
import numpy as np
import pickle

from GNN.layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        """
        :param x: embedding weights
        :param adj: relationship matrix 1/0
        :return:
        """
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x

#
# class ObjectDecoder(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, embeddings, graph_dropout, k):
#         super(ObjectDecoder, self).__init__()
#         self.k = k
#         self.decoder = DecoderRNN2(hidden_size, output_size, embeddings, graph_dropout)
#         self.max_decode_steps = 2
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input, input_hidden, vocab, vocab_rev, decode_steps_t, graphs):
#         all_outputs, all_words = [], []
#
#         decoder_input = torch.tensor([vocab_rev['<s>']] * input.size(0)).cuda()
#         decoder_hidden = input_hidden.unsqueeze(0)
#         torch.set_printoptions(profile="full")
#
#         for di in range(self.max_decode_steps):
#             ret_decoder_output, decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, input,
#                                                                               graphs)
#
#             if self.k == 1:
#                 all_outputs.append(ret_decoder_output)
#
#                 dec_objs = []
#                 for i in range(decoder_output.shape[0]):
#                     dec_probs = F.softmax(ret_decoder_output[i][graphs[i]], dim=0)
#                     idx = dec_probs.multinomial(1)
#                     graph_list = graphs[i].nonzero().cpu().numpy().flatten().tolist()
#                     assert len(graph_list) == dec_probs.numel()
#                     dec_objs.append(graph_list[idx])
#                 topi = torch.LongTensor(dec_objs).cuda()
#
#                 # dec_probs = self.softmax(decoder_output)
#                 # topi = dec_probs.multinomial(num_samples=1)
#                 # topi = self.softmax(decoder_output).topk(1)[1]
#
#                 decoder_input = topi.squeeze().detach()
#
#                 all_words.append(topi)
#             else:
#                 topv, topi = decoder_output.topk(self.k)
#                 topv = self.softmax(topv)
#                 topv = topv.cpu().numpy()
#                 topi = topi.cpu().numpy()
#                 cur_objs = []
#
#                 for i in range(graphs.size(0)):
#                     cur_obj = np.random.choice(topi[i].reshape(-1), p=topv[i].reshape(-1))
#                     cur_objs.append(cur_obj)
#
#                 decoder_input = torch.LongTensor(cur_objs).cuda()
#                 all_words.append(decoder_input)
#                 all_outputs.append(decoder_output)
#
#         return torch.stack(all_outputs), torch.stack(all_words)
#
#     def flatten_parameters(self):
#         self.encoder.gru.flatten_parameters()
#         self.decoder.gru.flatten_parameters()
#
#
# class KGA2C(nn.Module):
#     def __init__(self, params, templates, max_word_length, vocab_act,
#                  vocab_act_rev, input_vocab_size, gat=True):
#         super(KGA2C, self).__init__()
#         self.templates = templates
#         self.gat = gat
#         self.max_word_length = max_word_length
#         self.vocab = vocab_act
#         self.vocab_rev = vocab_act_rev
#         self.batch_size = params['batch_size']
#         self.action_emb = nn.Embedding(len(vocab_act), params['embedding_size'])
#         ###
#         self.state_emb = nn.Embedding(input_vocab_size, params['embedding_size'])
#         ###
#         self.action_drqa = ActionDrQA(input_vocab_size, params['embedding_size'],
#                                       params['batch_size'], params['recurrent'])
#         ###
#         self.state_gat = StateNetwork(params['gat_emb_size'],
#                                       vocab_act, params['embedding_size'],
#                                       params['dropout_ratio'], params['tsv_file'])
#         ###
#         self.template_enc = EncoderLSTM(input_vocab_size, params['embedding_size'],
#                                         int(params['hidden_size'] / 2),
#                                         params['padding_idx'], params['dropout_ratio'],
#                                         self.action_emb)
#         if not self.gat:
#             self.state_fc = nn.Linear(110, 100)
#         else:
#             self.state_fc = nn.Linear(210, 100)
#         self.decoder_template = DecoderRNN(params['hidden_size'], len(templates))
#         self.decoder_object = ObjectDecoder(50, 100, len(self.vocab.keys()),
#                                             self.action_emb, params['graph_dropout'],
#                                             params['k_object'])
#         self.softmax = nn.Softmax(dim=1)
#         self.critic = nn.Linear(100, 1)
#
#     def get_action_rep(self, action):
#         action = str(action)
#         decode_step = action.count('OBJ')
#         action = action.replace('OBJ', '')
#         action_desc_num = 20 * [0]
#
#         for i, token in enumerate(action.split()[:20]):
#             short_tok = token[:self.max_word_length]
#             action_desc_num[i] = self.vocab_rev[short_tok] if short_tok in self.vocab_rev else 0
#
#         return action_desc_num, decode_step
#
#     # important!!!
#     def forward_next(self, obs, graph_rep):
#         o_t, h_t = self.action_drqa.forward(obs)
#         g_t = self.state_gat.forward(graph_rep)
#         state_emb = torch.cat((g_t, o_t), dim=1)
#         state_emb = F.relu(self.state_fc(state_emb))
#         value = self.critic(state_emb)
#         return value
#
#     def forward(self, obs, scores, graph_rep, graphs):
#         '''
#         :param obs: The encoded ids for the textual observations (shape 4x300):
#         The 4 components of an observation are: look - ob_l, inventory - ob_i, response - ob_r, and prev_action.
#
#         :type obs: ndarray
#
#         '''
#         batch = self.batch_size
#         # print('obs', obs.shape)
#         # print('graphs', graphs.shape)
#         o_t, h_t = self.action_drqa.forward(obs)
#
#         src_t = []
#
#         for scr in scores:
#             # fist bit encodes +/-
#             if scr >= 0:
#                 cur_st = [0]
#             else:
#                 cur_st = [1]
#             cur_st.extend([int(c) for c in '{0:09b}'.format(abs(scr))])
#             src_t.append(cur_st)
#
#         src_t = torch.FloatTensor(src_t).cuda()
#
#         if not self.gat:
#             state_emb = torch.cat((o_t, src_t), dim=1)
#         else:
#             g_t = self.state_gat.forward(graph_rep)
#             state_emb = torch.cat((g_t, o_t, src_t), dim=1)
#         state_emb = F.relu(self.state_fc(state_emb))
#         det_state_emb = state_emb.clone()  # .detach()
#         value = self.critic(det_state_emb)
#
#         decoder_t_output, decoder_t_hidden = self.decoder_template(state_emb, h_t)  # torch.zeros_like(h_t))
#
#         templ_enc_input = []
#         decode_steps = []
#
#         topi = self.softmax(decoder_t_output).multinomial(num_samples=1)
#         # topi = decoder_t_output.topk(1)[1]#self.params['k'])
#
#         for i in range(batch):
#             # print(topi[i].squeeze().detach().item())
#             templ, decode_step = self.get_action_rep(self.templates[topi[i].squeeze().detach().item()])
#             # print(templ, decode_step)
#             templ_enc_input.append(templ)
#             decode_steps.append(decode_step)
#
#         decoder_o_input, decoder_o_hidden_init0, decoder_o_enc_oinpts = self.template_enc.forward(
#             torch.tensor(templ_enc_input).cuda().clone())
#
#         decoder_o_output, decoded_o_words = self.decoder_object.forward(decoder_o_hidden_init0.cuda(),
#                                                                         decoder_t_hidden.squeeze_(0).cuda(), self.vocab,
#                                                                         self.vocab_rev, decode_steps, graphs)
#
#         return decoder_t_output, decoder_o_output, decoded_o_words, topi, value, decode_steps  # decoder_t_output#template_mask
#
#     def clone_hidden(self):
#         self.action_drqa.clone_hidden()
#
#     def restore_hidden(self):
#         self.action_drqa.restore_hidden()
#
#     def reset_hidden(self, done_mask_tt):
#         self.action_drqa.reset_hidden(done_mask_tt)
#

class GraphNN(nn.Module):
    def __init__(self, gat_emb_size, embedding_size, dropout_ratio, json_file, gat_output_size, type):
        super(GraphNN, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio
        self.gat_emb_size = gat_emb_size
        self.gat_output_size = gat_output_size
        self.type = type

        self.gat = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        if os.path.exists(json_file):
            self.vocab_kge = self.load_vocab_kge(json_file)  # node -> id
        else:
            print('Did not find ', json_file)
            raise FileNotFoundError

        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((len(self.vocab_kge), self.embedding_size)),freeze=False)  # empty embedding all 0s => 362 * 50
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, self.gat_output_size)  # what is 3?

    # def init_state_ent_emb(self, emb_size):
    #     embeddings = torch.zeros((len(self.vocab_kge), emb_size))
    #     print('embeddings', embeddings.shape)
    #     for i in range(len(self.vocab_kge)):
    #         graph_node_text = self.vocab_kge[i].split('_')
    #         graph_node_ids = []
    #         for w in graph_node_text:
    #             print('w', w)
    #             if w in self.vocab.keys():
    #                 print("self.vocab[w]", self.vocab[w])
    #                 if self.vocab[w] < len(self.vocab) - 2:
    #
    #                     graph_node_ids.append(self.vocab[w])
    #                 else:
    #                     graph_node_ids.append(1)
    #             else:
    #                 graph_node_ids.append(1)
    #         # print("graph_node_ids ", graph_node_ids)
    #         graph_node_ids = torch.LongTensor(graph_node_ids).cuda()
    #
    #         cur_embeds = self.pretrained_embeds(graph_node_ids)
    #
    #         cur_embeds = cur_embeds.mean(dim=0)
    #         embeddings[i, :] = cur_embeds
    #     self.state_ent_emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

    def load_vocab_kge(self, load_path):
        ent = {}
        with open(load_path, 'rb')as f:
            ent = pickle.load(f)
        return ent

    def forward(self, adj):
        # print(self.state_ent_emb.weight)
        out = []
        adj = torch.IntTensor(adj).cuda()
        x = self.gat.forward(self.state_ent_emb.weight, adj)
        if self.type == 'state':
            x = x.view(x.shape[0],-1)
        else:
            x = x.view(1,-1)
        out.append(x.unsqueeze_(0))
        out = torch.cat(out)
        ret = self.fc1(out)
        return ret


class StateNN(nn.Module):
    def __init__(self, state_size, state_output_size):
        super(StateNN, self).__init__()
        self.fcx = nn.Linear(state_size, state_output_size)

    # def reset_hidden(self, done_mask_tt):
    #     '''
    #     Reset the hidden state of episodes that are done.
    #
    #     :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
    #     :type done_mask_tt: Tensor of shape [BatchSize x 1]
    #
    #     '''
    #     self.h_look = done_mask_tt.detach() * self.h_look
    #     self.h_inv = done_mask_tt.detach() * self.h_inv
    #     self.h_ob = done_mask_tt.detach() * self.h_ob
    #     self.h_preva = done_mask_tt.detach() * self.h_preva
    #
    # def clone_hidden(self):
    #     ''' Makes a clone of hidden state. '''
    #     self.tmp_look = self.h_look.clone().detach()
    #     self.tmp_inv = self.h_inv.clone().detach()
    #     self.h_ob = self.h_ob.clone().detach()
    #     self.h_preva = self.h_preva.clone().detach()
    #
    # def restore_hidden(self):
    #     ''' Restores hidden state from clone made by clone_hidden. '''
    #     self.h_look = self.tmp_look
    #     self.h_inv = self.tmp_inv
    #     self.h_ob = self.h_ob
    #     self.h_preva = self.h_preva

    def forward(self, obs):
        '''
        :param obs: Encoded observation tokens.
        :type obs: np.ndarray of shape (Batch_Size x 4 x 300)

        '''
        x = F.relu(self.fcx(obs))
        return x