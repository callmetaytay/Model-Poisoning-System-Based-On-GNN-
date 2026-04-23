import os
import dgl
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from gcn import *
from metrics import *

th.manual_seed(1)


def run_gnn(args, data, return_history=False, verbose=True):
    train_g, test_g = data

    train_nid = th.arange(len(train_g.nodes()), dtype=th.int32)
    test_nid = th.arange(len(test_g.nodes()), dtype=th.int32)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )

    # Define model and optimizer
    model = GCN(args.in_feats, args.n_hidden, args.n_classes, args.n_layers, args.activation, args.batch_size,
                args.num_workers, args.dropout)
    model = model.to(args.device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # th.save(model.state_dict(), "w0.pth")
    if verbose:
        print("\nallready save")
        for name, param in model.named_parameters():
            print(f"name: {name} | shape: {param.shape} | example:\n{param.data}\n")

    # Training loop
    avg = 0
    iter_tput = []
    history = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        tic_step = time.time()
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

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                # print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                #    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))
            tic_step = time.time()

        toc = time.time()
        epoch_record = {
            "epoch": int(epoch),
            "epoch_time": round(float(toc - tic), 4),
            "train_acc": None,
            "test_acc": None,
            "is_final": False,
        }
        if verbose:
            print('Epoch %d, Time(s):%.4f' % (epoch, toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            train_acc, _ = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid,
                                    args.device)
            if verbose:
                print('Train Acc {:.4f}'.format(train_acc))

            test_acc, _ = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid,
                                   args.device)
            if verbose:
                print('Test Acc: {:.4f}'.format(test_acc))
            epoch_record["train_acc"] = round(float(train_acc), 4)
            epoch_record["test_acc"] = round(float(test_acc), 4)
        history.append(epoch_record)


    # saving_path = os.path.join(args.model_save_path,
    #                             '%s_%s_%s_%s.pth' % (args.setting, args.dataset, args.model, args.mode))
    # print("Finish training, save model to %s" % (saving_path))
    # th.save(model.state_dict(), saving_path)

    # finish training
    train_acc, _ = evaluate(model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, args.device)
    if verbose:
        print('Final Train Acc {:.4f}'.format(train_acc))

    test_acc, _ = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.device)
    if verbose:
        print('Final Test Acc {:.4f}'.format(test_acc))
    history.append(
        {
            "epoch": int(args.num_epochs),
            "epoch_time": 0.0,
            "train_acc": round(float(train_acc), 4),
            "test_acc": round(float(test_acc), 4),
            "is_final": True,
        }
    )

    # th.save(model.state_dict(), "final_w.pth")
    if verbose:
        print("\nallready save")
        for name, param in model.named_parameters():
            print(f"name: {name} | shape: {param.shape}")

    # fixed_param = [param.detach().clone() for _, param in model.named_parameters()]

    if return_history:
        return train_acc, test_acc, history
    return train_acc, test_acc
    # return fixed_param
