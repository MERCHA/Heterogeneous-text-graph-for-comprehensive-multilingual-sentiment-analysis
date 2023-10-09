import argparse
import os
import pickle as pkl
import time

import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from utils.utils import *


accuracy_results = []


def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GCN",
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='require early stopping.')
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices = ['mr', 'amazon'],
                        help='dataset to train')

    args, _ = parser.parse_known_args()
    return args

args = get_citation_args()

# Load data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(args.dataset)
print(features.shape)
features = sp.identity(features.shape[0])
print(features.shape)
features = preprocess_features(features)
print(features.shape)


def pre_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

adjdense = torch.from_numpy(pre_adj(adj).A.astype(np.float32))

def construct_graph(adjacency):
    g = DGLGraph()
    adj = pre_adj(adjacency)
    g.add_nodes(adj.shape[0])
    g.add_edges(adj.row,adj.col)
    adjdense = adj.A
    adjd = np.ones((adj.shape[0]))
    for i in range(adj.shape[0]):
        adjd[i] = adjd[i] * np.sum(adjdense[i,:])
    weight = torch.from_numpy(adj.data.astype(np.float32))
    g.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
    g.edata['w'] = weight

    if args.cuda:
        g.to(torch.device('cuda:0'))

    return g

class SimpleConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(SimpleConv, self).__init__()
        if args.cuda:
            g = g.to(torch.device('cuda:0'))
        self.graph = g
        self.activation = activation
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        self.feat_drop = feat_drop

    def forward(self, feat):
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.update_all(fn.src_mul_edge(src='h', edge='w', out='m'), fn.sum(msg='m',out='h'))
        rst = g.ndata['h']
        rst = self.activation(rst)
        return rst


"""Slightly deep GCN"""
class Classifer(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super(Classifer, self).__init__()
        self.linear = nn.Linear(100,num_classes)
        self.act = F.softmax
        self.GCN = conv
        self.gcn1 = self.GCN(g,input_dim, 200, torch.tanh)
        self.gcn2 = self.GCN(g, 200, 100, torch.tanh)

    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)
        x = self.act(self.linear(x))
        return x


# """ GCN"""
# class Classifer(nn.Module):
#     def __init__(self,g,input_dim,num_classes,conv):
#         super(Classifer, self).__init__()
#         self.GCN = conv
#         self.gcn1 = self.GCN(g,input_dim, 200, F.relu)
#         self.gcn2 = self.GCN(g, 200, num_classes, F.relu)

#     def forward(self, features):
#         x = self.gcn1(features)
#         self.embedding = x
#         x = self.gcn2(x)
#         return x

g = construct_graph(adj)
# Define placeholders
t_features = torch.from_numpy(features.astype(np.float32))
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
support = [preprocess_adj(adj)]
num_supports = 1
t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

if args.model == 'GCN':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SimpleConv)
else:
    raise NotImplemented
# support has only one element, support[0] is adjacency
if args.cuda and torch.cuda.is_available():
    t_features = t_features.cuda()
    t_y_train = t_y_train.cuda()
    t_train_mask = t_train_mask.cuda()
    tm_train_mask = tm_train_mask.cuda()
    model = model.cuda()

print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def evaluate(features, labels, mask):
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features).cpu()
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

val_losses = []

# Train model
for epoch in range(args.epochs):

    t = time.time()

    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > args.early_stopping and val_losses[-1] > np.mean(val_losses[-(args.early_stopping+1):-1]):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))


accuracy_results.append(format(test_acc, '.5f'))
print(accuracy_results)
