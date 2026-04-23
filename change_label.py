import argparse

import dgl
import torch as th
import torch.nn as nn

from gcn import GCN
from load_data import load_graphgallery_data


def build_args():
    parser = argparse.ArgumentParser(description="Generate a label-poisoned graph.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--checkpoint", default="w0.pth")
    parser.add_argument("--output", default="updated_label_g.bin")
    parser.add_argument("--n-poison-nodes", type=int, default=300)
    parser.add_argument(
        "--selection",
        choices=["degree", "random"],
        default="degree",
        help="How to choose poisoned nodes.",
    )
    parser.add_argument(
        "--target",
        choices=["highest_wrong", "cyclic"],
        default="highest_wrong",
        help="How to assign poisoned labels.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def build_model(graph, n_classes: int, checkpoint: str, device: th.device) -> GCN:
    model = GCN(
        in_feats=int(graph.ndata["features"].shape[1]),
        n_hidden=16,
        n_classes=n_classes,
        n_layers=2,
        activation=nn.ReLU(),
        batch_size=256,
        num_workers=0,
        dropout=0.5,
    ).to(device)
    state_dict = th.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def full_graph_logits(model: GCN, graph, features: th.Tensor) -> th.Tensor:
    hidden = features
    for layer_index, layer in enumerate(model.layers):
        hidden = layer(graph, hidden)
        if layer_index != len(model.layers) - 1:
            hidden = model.activation(hidden)
    return hidden


def select_nodes(graph, n_poison_nodes: int, selection: str, seed: int) -> th.Tensor:
    total_nodes = graph.number_of_nodes()
    count = min(total_nodes, n_poison_nodes)
    if selection == "degree":
        degrees = graph.in_degrees()
        return th.argsort(degrees, descending=True)[:count]

    generator = th.Generator()
    generator.manual_seed(seed)
    return th.randperm(total_nodes, generator=generator)[:count]


def choose_poisoned_labels(
    logits: th.Tensor,
    labels: th.Tensor,
    selected_nodes: th.Tensor,
    target_mode: str,
    n_classes: int,
) -> th.Tensor:
    poisoned_labels = labels.clone()
    if target_mode == "cyclic":
        poisoned_labels[selected_nodes] = (poisoned_labels[selected_nodes] + 1) % n_classes
        return poisoned_labels

    probabilities = th.softmax(logits[selected_nodes], dim=1)
    probabilities[th.arange(len(selected_nodes)), labels[selected_nodes]] = -1.0
    poisoned_labels[selected_nodes] = th.argmax(probabilities, dim=1)
    return poisoned_labels


def evaluate_accuracy(logits: th.Tensor, labels: th.Tensor) -> float:
    predictions = th.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def main():
    args = build_args()
    th.manual_seed(args.seed)

    graph, n_classes = load_graphgallery_data(args.dataset)
    device = th.device("cpu")
    graph = graph.to(device)
    model = build_model(graph, n_classes, args.checkpoint, device)

    original_labels = graph.ndata["labels"].clone()
    selected_nodes = select_nodes(graph, args.n_poison_nodes, args.selection, args.seed)

    with th.no_grad():
        clean_logits = full_graph_logits(model, graph, graph.ndata["features"])
        clean_accuracy = evaluate_accuracy(clean_logits, original_labels)
        poisoned_labels = choose_poisoned_labels(
            clean_logits,
            original_labels,
            selected_nodes,
            args.target,
            n_classes,
        )
        poison_accuracy = evaluate_accuracy(clean_logits, poisoned_labels)

    graph.ndata["labels"] = poisoned_labels
    dgl.save_graphs(args.output, [graph])

    changed = int((poisoned_labels != original_labels).sum().item())
    print(f"Saved poisoned graph to {args.output}")
    print(f"Selected nodes: {len(selected_nodes)}")
    print(f"Changed labels: {changed}")
    print(f"Clean-label accuracy under fixed model: {clean_accuracy:.4f}")
    print(f"Poison-label accuracy under fixed model: {poison_accuracy:.4f}")


if __name__ == "__main__":
    main()
