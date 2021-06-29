

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator



class AGGCN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_emb_dim: int,
                 feature_dim: int,
                 tree_prop: int = 1,
                 tree_dropout: float=0.0,
                 aggcn_heads: int=4,
                 aggcn_sublayer_first: int=2,
                 aggcn_sublayer_second: int=4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AGGCN, self).__init__(vocab, regularizer)

        self.in_dim = span_emb_dim
        self.mem_dim = span_emb_dim

        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.num_layers = tree_prop

        self.layers = nn.ModuleList()

        self.heads = aggcn_heads
        self.sublayer_first = aggcn_sublayer_first
        self.sublayer_second = aggcn_sublayer_second

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_first))
                self.layers.append(GraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_second))
            else:
                self.layers.append(MultiGraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

        # mlp output layer
        in_dim = span_emb_dim
        mlp_layers = [nn.Linear(in_dim, feature_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*mlp_layers)
        # initializer(self)

    # adj: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, emb_dim)
    # text_mask: (batch, sequence)
    def forward(self, adj, text_embeddings, text_mask):

        gcn_inputs = self.input_W_G(text_embeddings)
        text_mask = text_mask.unsqueeze(-2)
        layer_list = []
        outputs = gcn_inputs
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        for i in range(len(self.layers)):
            if i < 2:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)
            else:
                attn_tensor = self.attn(outputs, outputs, text_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)

        outputs = self.out_mlp(dcgcn_output)
        return outputs


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, tree_dropout, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(tree_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, tree_dropout, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(tree_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))


    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out




def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn



