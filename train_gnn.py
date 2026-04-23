import os
# from collections.abc import generator

import dgl
import time
import numpy as np
import torch as th
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.cuda import device

# from change_point import global_hook_enable
from gcn import *
from metrics import *
from load_data import *

th.manual_seed(1)


def train_gnn(minmin,args, train_g, test_g):

    # # the statistical distribution of the feature of train graph and test graph
    # train_node_feats = train_g.ndata['features'].cpu().numpy()
    # df = pd.DataFrame(train_node_feats, columns=[f'feat_{i}' for i in range(train_node_feats.shape[1])])
    # print("the statistical distribution of train_g's features")
    # print(df.describe())
    # test_node_feats = test_g.ndata['features'].cpu().numpy()
    # df = pd.DataFrame(test_node_feats, columns=[f'feat_{i}' for i in range(test_node_feats.shape[1])])
    # print("the statistical distribution of test_g's features")
    # print(df.describe())


    # random 10 nodes' features in train_g and test_g
    # th.set_printoptions(sci_mode=False)
    # train_g_nodes = train_g.num_nodes()
    # train_g_rand_indices = th.randint(0, train_g_nodes, (10, ), generator=th.manual_seed(args.seed))
    # print("train_g's features: ", train_g.ndata['features'][train_g_rand_indices])
    #
    # test_g_nodes = test_g.num_nodes()
    # test_g_rand_indices = th.randint(0, test_g_nodes, (10,), generator=th.manual_seed(args.seed))
    # print("test_g's features: ", test_g.ndata['features'][test_g_rand_indices])
    # th.set_printoptions(sci_mode=True)

    train_g.create_formats_()
    test_g.create_formats_()

    # num_workers = 0
    safe_num_workers = 0 if args.device.type == 'cuda' else args.num_workers

    train_nid = th.arange(len(train_g.nodes()), device=args.device, dtype=th.int32)
    test_nid = th.arange(len(test_g.nodes()), device=args.device, dtype=th.int32)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=safe_num_workers
    )

    # Define model and optimizer
    model = GCN(args.in_feats, args.n_hidden, args.n_classes, args.n_layers, args.activation, args.batch_size,
                args.num_workers, args.dropout)
    model = model.to(args.device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.num_epochs):
        for step, (_, seeds, blocks) in enumerate(dataloader):
            blocks = [block.int().to(args.device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels'].to(device=args.device, dtype=th.long)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    print(f'Epoch{minmin}')
    train_acc, _ = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, args.device)
    print('Final Train Acc {:.4f}'.format(train_acc))

    test_acc, _ = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
    print('Final Test Acc {:.4f}'.format(test_acc))

    current_params = list(model.parameters())
    return current_params, test_acc
