from graphgallery.datasets import NPZDataset

def load_graphgallery_data(datapath, dataset):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset, verbose=False)
    graph = data.graph

    # print(graph)
    # print(graph.node_attr)
    # print(graph.label)

    adj = graph.adj_matrix
    features = graph.node_attr
    label = graph.label

    if dataset in ['blogcatalog', 'flickr']:
        label = [x - 1 for x in label]

    return adj, features , label