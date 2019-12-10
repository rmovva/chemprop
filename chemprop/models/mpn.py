from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.attn_hidden_size = args.attn_hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size # + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        self.bondKeys = True # flag for whether or not 
        self.usingGru = True # flag for whether or not 
        if (self.bondKeys):
            self.featLen = self.hidden_size + self.bond_fdim
        else:
            self.featLen = self.hidden_size
        
        self.norm1 = nn.LayerNorm(self.hidden_size)
        # self.attn_hidden_size = self.hidden_size / 2
        self.heads = 1 #TO-DO: HOW DO I DO HYPER PARAMETER OPTIMIZAITON W THIS VAL?. also todo implement multihead.
        self.tokeys    = nn.Linear(self.featLen, self.attn_hidden_size * self.heads, bias=False)
        self.toqueries = nn.Linear(self.hidden_size, self.attn_hidden_size * self.heads, bias=False)
        self.tovalues  = nn.Linear(self.hidden_size, self.attn_hidden_size * self.heads, bias=False)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1)
        self.ht = None #hidden state of gru, hidden x 1

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_atoms/num_bonds x hidden_size
        
        scale = np.sqrt(self.attn_hidden_size) # hyperparameter for pre softmax normalization,

        # hyperparameter search with scale = 1?
        # Message passing
        self.ht = f_atoms
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                # try muliplying softmax by count of # of atoms
                
                # filtered_messages = index_select_ND(self.tovalues(message), a2a)
                # w = self.toqueries(message) @ self.tokeys(message).T / np.sqrt(self.hidden * self.heads) #num_atoms x num_atoms
                # above normalized to help softmax converge (not sure if neccessary but mentioned in peter bloems post) 
                
                # add masking (incomplete)
                # w = F.softmax(w, dim = 1) # num_atoms x num_atoms
                # message_weighted = w @ self.tovalues(message) # num_atoms x (self.hidden_size * self.heads), basically weight messages
                # meaningless global weighting without respect to position embedding; need to filter to local only
                filtered_key_messages = None
                if (self.bondKeys): #if want bond features in the key
                    nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                    nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                    nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                    filtered_key_messages = self.tokeys(nei_message) # num_atoms x max_num_bonds x hidden
                else:
                    filtered_key_messages = index_select_ND(self.tokeys(message), a2a) # num_atoms x max_num_bonds x hidden
                
                # manual dot product done by duplicating query max_num_bonds times and elementwise mult w keys
                duplicated_query = self.toqueries(message).unsqueeze(dim=1).repeat(1,a2a.shape[1],1) # num_atoms x max_num_bonds x hidden
                queryxKey = filtered_key_messages * duplicated_query / scale # num_atoms x max_num_bonds x hidden
                queryxKey = queryxKey.sum(dim = 2) # num_atoms x max_num_bonds x 1
                w = F.softmax(queryxKey, dim = 1) # num_atoms x max_num_bonds

                # filtered means only the neighbors
                filtered_value_messages = index_select_ND(self.tovalues(message), a2a) # num_atoms x max_num_bonds x hidden

                duplicated_attn_weights = w.unsqueeze(dim=2).repeat(1,1,filtered_value_messages.shape[2]) # num_atoms x max_num_bonds x hidden
                message_weighted = (duplicated_attn_weights * filtered_value_messages).sum(dim = 1) # num_atoms x hidden
            else:
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
                message_weighted = message

            # message_weighted = self.W_h(message_weighted)
            message_weighted = self.act_func(input + message_weighted)  # num_bonds x hidden_size
            if(self.usingGru):
                message_weighted = self.norm1(message_weighted)  # num_bonds x hidden
                h, _ = self.gru(message_weighted.unsqueeze(dim=0), message.unsqueeze(dim=0)) # message is the hidden state in the gru. o = h so we discard a value at random
                message = h[0]
                
                if(not torch.eq(h,  _).all()):
                    print("Aayush was wrong about the gru. Validation.")
                # log h and _ being equal
            else:
                message_weighted = self.dropout_layer(message_weighted)  # num_bonds x hidden
                message = message_weighted
            
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output
