from itertools import chain, combinations
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
import numpy as np
import itertools
import scipy.special
from time import time
import dgl
import torch

# def powerset(iterable, order = None):
#     n_node_subset = iterable.num_nodes()
#     all_subgraphs = []
#
#     for r in range(1, n_node_subset + 1):
#         for nodes in itertools.combinations(range(n_node_subset), r):
#             node_ids = torch.tensor(nodes)
#             subg = dgl.node_subgraph(iterable, node_ids)
#             all_subgraphs.append(subg)
#
#     return all_subgraphs
def powerset(iterable, order=None):
    if order is None:
        order = len(iterable)
    return {r:list(combinations(iterable, r)) for r in range(order+1)}

def convert_index_to_categorical(indices, d):
    a,b = indices.shape
    if b ==0:
        return np.zeros((a,d))
    else:
        indices = np.reshape(indices,[-1])
        cats = to_categorical(indices, num_classes=d)
        cats = np.reshape(cats,(a,b,d))
        return np.sum(cats,axis=-2)

# def localshapley(g, k):
#     # nunber of elements, including i
#     max_order = k + 1
#
#     # recipical of coefficients
#     coefficients = {}
#     for j in range(1, max_order+1):
#         binomial_coeff = scipy.special.binom(k, j-1)
#         denominator = float(k+1) * binomial_coeff
#         coefficient = 1.0 / denominator
#         coefficients[j] = coefficient
#
#     # construct collection of subsets for each feature
#     subsets = {}
#     complete_subset = {}
#
#     for node in g.nodes():
#         complete_subset = dgl.khop_in_subgraph(g, node, k)
#         subsets[node] = powerset(complete_subset, max_order-1)
#
#     positions_dict = {
#         (node, j, fill): []
#         for node in g.nodes()
#         for j in range(1, max_order + 1)
#         for fill in [0,1]
#     }
#
#     for i in subsets:
def localshapley(n_nodes, k):
    while k>= n_nodes:
        k -= 2
    max_order =k+1
    coefficients = {
        j:(1.0/scipy.special.binom(k, j-1))/float(k+1)
        for j in range(1, max_order+1)
    }

    subsets = {}
    for i in range(n_nodes):
        neighbor_indices = list(range(i-k//2, i)) + list(range(i+1,i+k//2+1))
        subset = list(np.array(neighbor_indices) % n_nodes)
        subsets[i] = powerset(subset, max_order - 1)

    positions_dict = {(i, j, fill): [] for i in range(n_nodes) for j in range(1, max_order + 1) for fill in [0, 1]}

    for i in subsets:
        one_pads = 1 - np.sum(
            to_categorical(np.array(range(i - k // 2, i + k // 2 + 1)) % n_nodes, num_classes=n_nodes), axis=0)
        for j in subsets[i]:
            for arr in subsets[i][j]:
                pos_excluded = np.sum(to_categorical(arr, num_classes=n_nodes), axis=0)
                pos_excluded += one_pads
                pos_included = pos_excluded + to_categorical(i, num_classes=n_nodes)
                positions_dict[(i, j + 1, 1)].append(pos_included)
                positions_dict[(i, j + 1, 0)].append(pos_excluded)

    keys = list(positions_dict.keys())
    values = [np.array(v) for v in positions_dict.values()]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    print('checking uniqueness...')
    positions, unique_inverse = np.unique(positions, axis=0, return_inverse=True)
    return positions_dict, key_to_idx, positions, coefficients, unique_inverse

def generate_subgraph_inputs_dgl(mask_vectors, x_full, full_graph):
    """
    mask_vectors: (n_samples, n_nodes) numpy array, 0/1掩码
    x_full: (n_nodes, d) 节点特征矩阵（numpy 或 torch）
    full_graph: dgl.DGLGraph
    返回：[(subgraph, x_sub)] 列表
    """
    subgraphs = []
    for mask in mask_vectors:
        node_ids = np.where(mask == 1)[0]
        if len(node_ids) == 0:
            node_ids = [0]  # 防止空子图
        sub_g = dgl.node_subgraph(full_graph, node_ids)
        x_sub = x_full[node_ids]
        subgraphs.append((sub_g, x_sub))
    return subgraphs

def predict_dgl(model, inputs):
    model.eval()
    outputs = []
    with torch.no_grad():
        for g_sub, x_sub in inputs:
            x_tensor = torch.tensor(x_sub, dtype=torch.float32)
            out = model(g_sub, x_tensor)  # 返回图级概率向量 shape=[1, C]
            probs = F.softmax(out, dim=-1)
            outputs.append(probs.cpu().numpy()[0])
    return np.array(outputs)


def explain_node_contributions_dgl(model, x, g, k):
    """
    model: 接受 (g_sub, x_sub) 的图分类模型
    x: 全图节点特征 numpy array，(n_nodes, d)
    g: DGLGraph 图结构
    k: 局部邻域大小
    """
    n_nodes = x.shape[0]
    pos_dict, key_to_idx, pos_masks, coeffs, unique_inv = localshapley(n_nodes, k)

    sub_inputs = generate_subgraph_inputs_dgl(pos_masks, x, g)
    prob = predict_dgl(model, [(g, x)])[0]
    label_one_hot = np.eye(len(prob))[np.argmax(prob)]

    f_vals = predict_dgl(model, sub_inputs)
    log_probs = np.log(f_vals + np.finfo(float).resolution)
    vals = np.sum(label_one_hot * log_probs, axis=1)

    key_to_val = {
        key: np.array([vals[unique_inv[idx]] for idx in key_to_idx[key]])
        for key in key_to_idx
    }

    phis = np.zeros(n_nodes)
    for i in range(n_nodes):
        phis[i] = np.sum([
            coeffs[j] * np.sum(key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)])
            for j in coeffs
        ])
    return phis




