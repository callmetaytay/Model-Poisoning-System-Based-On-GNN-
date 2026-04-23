import torch
import torch as th

from load_data import *
from argparse import Namespace

from run_gnn import *
import warnings
warnings.filterwarnings("ignore", message="Dataloader CPU affinity opt is not enabled")


# adj, features, label = load_graphgallery_data('./datasets', 'cora')
# print(adj, features, label)
# datapath = 'data/dataset/original/'
# adj, features, gcn_pred_sign = load_data(datapath, 'cora')

if __name__ == '__main__':

    # g,n_classes = load_graphgallery_data('cora')

    # data = np.load('datasets/modified_cora_graph.npz')
    # print(data.files)
    #
    # node_features = data['node_features']
    # node_labels = data['node_labels']
    # print(node_features.shape)
    # edges_src = data['edges_src']
    # edges_dst = data['edges_dst']
    #
    # g = dgl.graph((edges_src, edges_dst))
    #
    # g.ndata["features"] = torch.tensor(node_features, dtype=torch.float32)
    # g.ndata["labels"] = torch.tensor(node_labels, dtype=torch.long)

    graphs, _= dgl.load_graphs('updated_features_g.bin')
    g = graphs[0]



    in_featrue = g.ndata['features'].shape[1]

    train_g, test_g = split_train_test(g)

    train_g.create_formats_()
    test_g.create_formats_()

    run_data = train_g, test_g

    if th.cuda.is_available():
        device = th.device('cuda:0')
        num_workers = 4
    else:
        device = th.device('cpu')
        num_workers = 0

    args = Namespace(
        in_feats=1433,         # Cora 数据集中每个节点的特征维度为 1433
        n_hidden=16,           # 隐藏层的维度常设为 16，能提供良好的性能
        n_classes=7,           # Cora 数据集中有 7 个类别
        n_layers=2,            # 图卷积层的数量，通常为 2 层，足以捕获有效信息
        activation=nn.ReLU(),  # ReLU 激活函数
        batch_size=256,        # 批次大小，Cora 数据集的节点数为 2708，256 是常用的批次
        num_workers=num_workers,  # 按设备选择更稳的 DataLoader 配置
        dropout=0.5,            # Dropout 概率为 0.5，较为常见
        model = 'gcn',
        lr = 0.001,
        num_epochs = 300,
        log_every = 20,
        eval_every = 5,
        device=device,
    )

    train_acc, test_acc = run_gnn(args, run_data)

    # print(args.dataset, args.model, args.model, args.seed)
    # print(args.dataset, args.model, args.model)
    # print("train_acc:%.3f\n" % args.train_acc)
    # print("test_acc:%.3f\n" % args.test_acc)

