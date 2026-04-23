"""
FastAPI backend for the real node-poisoning experiment site.
"""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import dgl
except ModuleNotFoundError as exc:
    dgl = None
    DGL_IMPORT_ERROR = str(exc)
else:
    DGL_IMPORT_ERROR = None

try:
    if dgl is not None:
        from gcn import GCN
        from load_data import load_graphgallery_data
        from run_gnn import run_gnn
    else:
        GCN = None
        load_graphgallery_data = None
        run_gnn = None
except ModuleNotFoundError as exc:
    GCN = None
    load_graphgallery_data = None
    run_gnn = None
    DGL_IMPORT_ERROR = str(exc)

ROOT_DIR = Path(__file__).resolve().parent
ATTACK_SCRIPT = ROOT_DIR / "change_label.py"

app = FastAPI(title="GNN安全检测API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: GCN | None = None
GRAPH = None
CURRENT_GRAPH = None
ARGS: Namespace | None = None
BASELINE_METRICS: dict[str, float] = {}
GRAPH_LAYOUT: dict[int, tuple[float, float]] = {}
ATTACK_HISTORY: list[dict[str, Any]] = []
CURRENT_ATTACK_STATE: dict[str, Any] = {
    "poisoned_nodes": [],
}


class AttackRequest(BaseModel):
    n_poison_nodes: int = Field(gt=0)
    lr: float = Field(gt=0)
    num_epochs: int = Field(gt=0)


class GraphData(BaseModel):
    nodes: int
    edges: int
    features: int
    classes: int
    density: float


class MetricsResponse(BaseModel):
    robustness: float
    accuracy: float
    clean_accuracy: float
    poison_accuracy: float
    clean_train_accuracy: float
    poison_train_accuracy: float
    avg_confidence: float
    changed_fraction: float
    experiment_count: int


class NodeInfo(BaseModel):
    node_id: int
    original_label: int
    predicted_label: int
    is_poisoned: bool
    degree: int
    features_avg: float
    features_nonzero: int
    confidence: float


def compute_acc(pred: th.Tensor, labels: th.Tensor) -> th.Tensor:
    labels = labels.long()
    pred = pred.to(labels.device)
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def _require_ready() -> tuple[GCN, Any, Namespace]:
    if DGL_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=503,
            detail=f"后端缺少依赖，无法加载 GNN 模型: {DGL_IMPORT_ERROR}",
        )
    if MODEL is None or GRAPH is None or ARGS is None:
        raise HTTPException(status_code=503, detail="模型或图数据尚未准备完成")
    return MODEL, GRAPH, ARGS


def _active_graph():
    return CURRENT_GRAPH if CURRENT_GRAPH is not None else GRAPH


def _predict(graph) -> tuple[th.Tensor, np.ndarray]:
    model, _, args = _require_ready()
    model.eval()
    with th.no_grad():
        logits = model.inference(graph, graph.ndata["features"], args.device)
        probs = th.softmax(logits, dim=1)
    return logits, probs.cpu().numpy()


def _evaluate_graph(graph, labels: th.Tensor | None = None) -> dict[str, Any]:
    logits, probs = _predict(graph)
    used_labels = labels if labels is not None else graph.ndata["labels"]
    accuracy = float(compute_acc(logits, used_labels))
    predictions = th.argmax(logits, dim=1)
    return {
        "accuracy": accuracy,
        "predictions": predictions.cpu(),
        "probabilities": probs,
        "avg_confidence": float(np.max(probs, axis=1).mean()),
    }


def _build_layout(graph) -> dict[int, tuple[float, float]]:
    nx_g = graph.to_networkx().to_undirected()
    sampled_nodes = list(range(min(120, graph.number_of_nodes())))
    subgraph = nx_g.subgraph(sampled_nodes).copy()
    return nx.spring_layout(subgraph, seed=42, k=0.35, iterations=60)


def _compute_robustness(clean_accuracy: float, poison_accuracy: float) -> float:
    if clean_accuracy <= 1e-6:
        return 0.0
    return round(max(0.0, min(1.0, poison_accuracy / clean_accuracy)), 4)


def _current_metrics() -> dict[str, Any]:
    latest = ATTACK_HISTORY[-1] if ATTACK_HISTORY else None
    if latest is None:
        return {
            "robustness": 1.0,
            "accuracy": BASELINE_METRICS["clean_accuracy"],
            "clean_accuracy": BASELINE_METRICS["clean_accuracy"],
            "poison_accuracy": BASELINE_METRICS["clean_accuracy"],
            "clean_train_accuracy": BASELINE_METRICS["clean_accuracy"],
            "poison_train_accuracy": BASELINE_METRICS["clean_accuracy"],
            "avg_confidence": BASELINE_METRICS["avg_confidence"],
            "changed_fraction": 0.0,
            "experiment_count": 0,
        }

    return {
        "robustness": latest["robustness"],
        "accuracy": latest["poison_accuracy"],
        "clean_accuracy": latest["clean_accuracy"],
        "poison_accuracy": latest["poison_accuracy"],
        "clean_train_accuracy": latest["clean_train_accuracy"],
        "poison_train_accuracy": latest["poison_train_accuracy"],
        "avg_confidence": latest["avg_confidence"],
        "changed_fraction": latest["changed_fraction"],
        "experiment_count": len(ATTACK_HISTORY),
    }


def _build_comparison_data(clean_accuracy: float, poison_accuracy: float) -> dict[str, list[float | str]]:
    return {
        "labels": ["训练集 Train Acc", "中毒测试集 Test Acc"],
        "accuracy": [round(clean_accuracy, 4), round(poison_accuracy, 4)],
    }


def _build_epoch_chart(history: list[dict[str, float]], metric_key: str) -> dict[str, list[float | int]]:
    filtered = [item for item in history if item.get(metric_key) is not None]
    return {
        "epochs": [item["epoch"] for item in filtered],
        "values": [item[metric_key] for item in filtered],
    }


def _build_split_indices(total_nodes: int, seed: int = 2025, prop: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(total_nodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(total_nodes * prop)
    train_index = np.sort(indices[:split])
    test_index = np.sort(indices[split:])
    return train_index, test_index


def _build_training_args(lr: float, num_epochs: int) -> Namespace:
    _, graph, _ = _require_ready()
    return Namespace(
        in_feats=int(graph.ndata["features"].shape[1]),
        n_hidden=16,
        n_classes=int(th.max(graph.ndata["labels"]).item()) + 1,
        n_layers=2,
        activation=nn.ReLU(),
        batch_size=256,
        num_workers=0,
        dropout=0.5,
        model="gcn",
        lr=lr,
        num_epochs=num_epochs,
        log_every=20,
        eval_every=5,
        device=th.device("cpu"),
    )


def _run_real_attack_script(n_poison_nodes: int) -> tuple[Any, str, str]:
    with tempfile.TemporaryDirectory(prefix="poison_node_") as tmpdir:
        output_path = Path(tmpdir) / "node_graph.bin"
        command = [
            sys.executable,
            str(ATTACK_SCRIPT),
            "--dataset",
            "cora",
            "--checkpoint",
            "w0.pth",
            "--n-poison-nodes",
            str(n_poison_nodes),
            "--selection",
            "degree",
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"真实攻击脚本执行失败:\n{completed.stderr or completed.stdout}",
            )
        graphs, _ = dgl.load_graphs(str(output_path))
        return graphs[0].int(), completed.stdout, " ".join(command)


def _run_training_metrics(clean_graph, poisoned_graph, lr: float, num_epochs: int) -> dict[str, Any]:
    if run_gnn is None:
        raise HTTPException(status_code=500, detail="run_gnn 模块未正确加载")

    train_index, test_index = _build_split_indices(clean_graph.number_of_nodes())

    clean_train_g = clean_graph.subgraph(train_index).int()
    poison_test_g = poisoned_graph.subgraph(test_index).int()

    clean_train_g.create_formats_()
    poison_test_g.create_formats_()

    args = _build_training_args(lr=lr, num_epochs=num_epochs)
    train_acc, poisoned_test_acc, history = run_gnn(
        args,
        (clean_train_g, poison_test_g),
        return_history=True,
        verbose=False,
    )

    return {
        "clean_train_accuracy": round(float(train_acc), 4),
        "poison_test_accuracy": round(float(poisoned_test_acc), 4),
        "history": history,
        "train_epochs": int(args.num_epochs),
    }


def _poisoned_nodes_from_graphs(base_graph, poisoned_graph) -> tuple[list[int], float]:
    changed_mask = poisoned_graph.ndata["labels"] != base_graph.ndata["labels"]
    changed_nodes = th.nonzero(changed_mask, as_tuple=False).squeeze(1).cpu().tolist()
    return changed_nodes, 0.0


@app.on_event("startup")
async def startup_event():
    global MODEL, GRAPH, CURRENT_GRAPH, ARGS, BASELINE_METRICS, GRAPH_LAYOUT

    if DGL_IMPORT_ERROR is not None:
        print(f"❌ 无法加载 DGL 相关依赖: {DGL_IMPORT_ERROR}")
        return

    print("🔄 正在加载GCN模型和数据...")
    GRAPH, n_classes = load_graphgallery_data("cora")
    GRAPH = GRAPH.int()
    CURRENT_GRAPH = None
    print(f"✓ 已加载Cora数据集: {GRAPH.number_of_nodes()} 节点, {GRAPH.number_of_edges()} 边")

    ARGS = Namespace(
        in_feats=int(GRAPH.ndata["features"].shape[1]),
        n_hidden=16,
        n_classes=n_classes,
        n_layers=2,
        activation=nn.ReLU(),
        batch_size=256,
        num_workers=0,
        dropout=0.5,
        device=th.device("cpu"),
        lr=0.001,
    )

    MODEL = GCN(
        ARGS.in_feats,
        ARGS.n_hidden,
        ARGS.n_classes,
        ARGS.n_layers,
        ARGS.activation,
        ARGS.batch_size,
        ARGS.num_workers,
        ARGS.dropout,
    ).to(ARGS.device)
    print("✓ 已初始化GCN模型")

    try:
        model_state = th.load("w0.pth", map_location="cpu")
        MODEL.load_state_dict(model_state)
        print("✓ 已加载预训练权重")
    except Exception:
        print("⚠️ 未找到预训练权重，使用随机初始化")

    baseline_eval = _evaluate_graph(GRAPH)
    BASELINE_METRICS = {
        "clean_accuracy": round(float(baseline_eval["accuracy"]), 4),
        "avg_confidence": round(float(baseline_eval["avg_confidence"]), 4),
    }
    GRAPH_LAYOUT = _build_layout(GRAPH)
    print("🚀 后端初始化完成")


@app.get("/api/health")
async def health_check():
    if DGL_IMPORT_ERROR is not None:
        return {
            "status": "degraded",
            "message": "后端已启动，但缺少 DGL 相关依赖",
            "dependency_error": DGL_IMPORT_ERROR,
        }
    return {"status": "ok", "message": "GNN安全检测API服务正常"}


@app.get("/api/graph-info", response_model=GraphData)
async def get_graph_info():
    _, _, _ = _require_ready()
    graph = _active_graph()
    nx_graph = graph.to_networkx().to_undirected()
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "features": int(graph.ndata["features"].shape[1]),
        "classes": int(th.max(graph.ndata["labels"]).item()) + 1,
        "density": float(nx.density(nx_graph)),
    }


@app.get("/api/graph-nodes")
async def get_graph_nodes():
    _, _, _ = _require_ready()
    graph = _active_graph()
    poisoned_set = set(CURRENT_ATTACK_STATE["poisoned_nodes"])
    nodes = []
    for node_id, (x_pos, y_pos) in GRAPH_LAYOUT.items():
        nodes.append(
            {
                "id": int(node_id),
                "x": float(x_pos),
                "y": float(y_pos),
                "label": int(graph.ndata["labels"][node_id]),
                "degree": int(graph.in_degrees(node_id)),
                "is_poisoned": node_id in poisoned_set,
            }
        )

    nx_g = graph.to_networkx().to_undirected()
    edges = []
    for source, target in nx_g.edges():
        if source in GRAPH_LAYOUT and target in GRAPH_LAYOUT:
            edges.append({"source": int(source), "target": int(target)})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/dashboard")
async def get_dashboard():
    _, _, _ = _require_ready()
    graph = _active_graph()
    metrics = _current_metrics()
    latest = ATTACK_HISTORY[-1] if ATTACK_HISTORY else None
    return {
        "metrics": metrics,
        "comparison": _build_comparison_data(metrics["clean_accuracy"], metrics["poison_accuracy"]),
        "latest_attack": latest,
        "graph_summary": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "features": int(graph.ndata["features"].shape[1]),
        },
    }


@app.post("/api/attack")
async def execute_attack(attack_config: AttackRequest):
    global CURRENT_GRAPH
    _, base_graph, _ = _require_ready()

    poisoned_graph, script_stdout, command = _run_real_attack_script(
        n_poison_nodes=attack_config.n_poison_nodes,
    )
    training_metrics = _run_training_metrics(
        base_graph,
        poisoned_graph,
        lr=attack_config.lr,
        num_epochs=attack_config.num_epochs,
    )

    clean_accuracy = training_metrics["clean_train_accuracy"]
    poison_accuracy = training_metrics["poison_test_accuracy"]
    poisoned_eval = _evaluate_graph(poisoned_graph)
    robustness = _compute_robustness(clean_accuracy, poison_accuracy)
    changed_nodes, feature_shift = _poisoned_nodes_from_graphs(base_graph, poisoned_graph)
    changed_fraction = round(len(changed_nodes) / poisoned_graph.number_of_nodes(), 4)

    CURRENT_GRAPH = poisoned_graph
    CURRENT_ATTACK_STATE["poisoned_nodes"] = changed_nodes

    attack_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_name": "修改节点中毒",
        "total_poisoned": int(len(changed_nodes)),
        "poisoned_nodes": changed_nodes[:50],
        "clean_accuracy": clean_accuracy,
        "poison_accuracy": poison_accuracy,
        "clean_train_accuracy": training_metrics["clean_train_accuracy"],
        "poison_train_accuracy": training_metrics["clean_train_accuracy"],
        "accuracy_drop": round(clean_accuracy - poison_accuracy, 4),
        "robustness": robustness,
        "avg_confidence": round(float(poisoned_eval["avg_confidence"]), 4),
        "changed_fraction": changed_fraction,
        "feature_shift": round(feature_shift, 6),
        "train_epochs": training_metrics["train_epochs"],
        "lr": attack_config.lr,
        "history": training_metrics["history"],
        "clean_train_curve": _build_epoch_chart(training_metrics["history"], "train_acc"),
        "poison_test_curve": _build_epoch_chart(training_metrics["history"], "test_acc"),
        "script_command": command,
        "script_stdout": script_stdout.strip().splitlines()[-8:],
    }
    ATTACK_HISTORY.append(attack_result)
    return {"status": "success", "data": attack_result, "history_size": len(ATTACK_HISTORY)}


@app.post("/api/reset")
async def reset_experiments():
    global CURRENT_GRAPH
    ATTACK_HISTORY.clear()
    CURRENT_GRAPH = None
    CURRENT_ATTACK_STATE["poisoned_nodes"] = []
    return {"status": "success", "message": "实验历史已清空"}


@app.get("/api/history")
async def get_history():
    return {"items": list(reversed(ATTACK_HISTORY[-20:]))}


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    return _current_metrics()


@app.get("/api/node-info/{node_id}", response_model=NodeInfo)
async def get_node_info(node_id: int):
    _, _, _ = _require_ready()
    graph = _active_graph()
    if node_id < 0 or node_id >= graph.number_of_nodes():
        raise HTTPException(status_code=404, detail="节点ID无效")

    node_eval = _evaluate_graph(graph)
    probabilities = node_eval["probabilities"][node_id]
    predicted_label = int(np.argmax(probabilities))
    features = graph.ndata["features"][node_id]

    return {
        "node_id": node_id,
        "original_label": int(graph.ndata["labels"][node_id]),
        "predicted_label": predicted_label,
        "is_poisoned": node_id in set(CURRENT_ATTACK_STATE["poisoned_nodes"]),
        "degree": int(graph.in_degrees(node_id)),
        "features_avg": round(float(features.mean()), 4),
        "features_nonzero": int(th.count_nonzero(features)),
        "confidence": round(float(np.max(probabilities)), 4),
    }


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    th.manual_seed(42)
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)
