import random
import numpy as np
import pickle as pkl
import sys
import scipy.sparse as sp

import torch
from pymetis import part_graph
import dgl


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def partition(adj_raw, n):
    """For the undirected graphs, adj_raw has stored each edge twice (node1, node2) and (node2, node1)"""
    nb_nodes = np.max(adj_raw) + 1
    adj_list = [[] for _ in range(nb_nodes)]
    for i, j in zip(adj_raw[0], adj_raw[1]):
        if i == j:
            continue
        adj_list[i].append(j)

    _, ss_labels = part_graph(nparts=n, adjacency=adj_list)

    return ss_labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return features


def load_data(dataset_str, seed, cluster_num, train_nodes_per_class):
    """
    Loads input data from ./data directory for cora, pubmed and citeseer

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # initial features: num_node * initial_dim
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    # sparse adj: the first row is the indices of source nodes, and the second row is the indices of target nodes
    # The target node is also the source node, since we perform on the undirected graph data
    adj_idx = []
    for node in graph.keys():
        adj_idx.extend([[node, node2] for node2 in graph[node]])    
    adj_idx = np.array(adj_idx).T
    
    # labels: num_node * 2
    # The first colum is the indices of nodes, and the second colum is the label index of the corresponding node
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    node_idx, label_idx = np.where(labels==1)
    labels = np.reshape(np.concatenate((node_idx, label_idx)), (2, -1)).T

    # Some nodes in citeseer are isolated, and they do not have labels
    nb_nodes = features.shape[0]
    if dataset_str == 'citeseer':
        tmp_labels = np.concatenate((np.reshape(range(nb_nodes), (-1, 1)), np.zeros((nb_nodes, 1))), 1).astype(int)
        for node, label in labels:
            tmp_labels[node, 1] = label
        labels = tmp_labels

    # Split train set, validation set and test set
    test_idx = test_idx_range
    train_val_idx = list(set(node_idx) - set(test_idx))
    train_val_label = labels[train_val_idx]
    train_idx = []
    val_idx = []
    # For each class, we sample 20 labeled instances to form the training set
    random.seed(seed)
    for idx in set(train_val_label[:, 1]):
        tmp = list(np.where(train_val_label[:, 1] == idx)[0])
        random.shuffle(tmp)
        train_idx.extend(list(train_val_label[tmp[: train_nodes_per_class], 0]))
        val_idx.extend(list(train_val_label[tmp[train_nodes_per_class:], 0]))
    
    # Perform METIS to derive graph community labels
    ss_label = partition(adj_idx, cluster_num)

    adj = dgl.graph((adj_idx.tolist()[0],adj_idx.tolist()[1]))
    adj = dgl.add_self_loop(adj)

    print('Number of nodes: ', nb_nodes)
    print('Number of edges: ', adj_idx.shape[1])
    print('Initial feature dimension: ', features.shape[1])
    print('Number of training nodes: ', len(train_idx))
    print('Number of validation nodes: ', len(val_idx))
    print('Number of testing nodes: ', len(test_idx))
    
    return adj, features, train_idx, val_idx, test_idx, labels, nb_nodes, ss_label

