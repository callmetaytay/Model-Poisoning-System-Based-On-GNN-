import argparse

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gcn import GCN
from load_data import load_graphgallery_data


def build_args():
    parser = argparse.ArgumentParser(description="Generate a feature-poisoned graph.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--checkpoint", default="w0.pth")
    parser.add_argument("--output", default="updated_features_g.bin")
    parser.add_argument("--n-poison-nodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument(
        "--selection",
        choices=["random", "degree"],
        default="random",
        help="How to choose poisoned nodes.",
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


def accuracy(logits: th.Tensor, labels: th.Tensor) -> float:
    predictions = th.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def main():
    args = build_args()
    th.manual_seed(args.seed)

    graph, n_classes = load_graphgallery_data(args.dataset)
    device = th.device("cpu")
    graph = graph.to(device)
    model = build_model(graph, n_classes, args.checkpoint, device)

    labels = graph.ndata["labels"]
    original_features = graph.ndata["features"].clone()
    selected_nodes = select_nodes(graph, args.n_poison_nodes, args.selection, args.seed)

    lower_bound = original_features.min()
    upper_bound = original_features.max()
    poisoned_features = original_features.clone()

    for _ in range(args.steps):
        poisoned_features.requires_grad_(True)
        logits = full_graph_logits(model, graph, poisoned_features)
        loss = F.cross_entropy(logits[selected_nodes], labels[selected_nodes])
        gradients = th.autograd.grad(loss, poisoned_features)[0]

        with th.no_grad():
            update = th.zeros_like(poisoned_features)
            update[selected_nodes] = gradients[selected_nodes].sign() * args.step_size
            poisoned_features = poisoned_features + update
            poisoned_features = th.max(
                th.min(poisoned_features, original_features + args.epsilon),
                original_features - args.epsilon,
            )
            poisoned_features = poisoned_features.clamp(lower_bound, upper_bound)

    with th.no_grad():
        clean_logits = full_graph_logits(model, graph, original_features)
        poison_logits = full_graph_logits(model, graph, poisoned_features)

    graph.ndata["features"] = poisoned_features.detach()
    dgl.save_graphs(args.output, [graph])

    feature_shift = float((poisoned_features - original_features).abs().mean().item())
    print(f"Saved poisoned graph to {args.output}")
    print(f"Selected nodes: {len(selected_nodes)}")
    print(f"Average feature shift: {feature_shift:.6f}")
    print(f"Clean accuracy under fixed model: {accuracy(clean_logits, labels):.4f}")
    print(f"Poison accuracy under fixed model: {accuracy(poison_logits, labels):.4f}")


if __name__ == "__main__":
    main()
