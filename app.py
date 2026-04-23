import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="图神经中毒攻击实验平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f6f2e9;
            --card: rgba(255,255,255,0.82);
            --ink: #1e2430;
            --muted: #5f6773;
            --line: rgba(30,36,48,0.08);
            --accent: #c65f3f;
            --accent-dark: #8e3518;
            --teal: #1f7a76;
            --gold: #d49b2d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(212,155,45,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(31,122,118,0.16), transparent 26%),
                linear-gradient(180deg, #fbf7ef 0%, #f2ede3 100%);
            color: var(--ink);
        }

        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp li,
        .stMarkdown,
        .stText,
        .stCaption {
            color: var(--ink) !important;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #241b14 !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }

        [data-testid="stAlertContainer"] * {
            color: #17324a !important;
        }

        [data-testid="stSidebar"] * {
            color: #eef2f7 !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #f8fbff !important;
            font-weight: 700 !important;
        }

        [data-testid="stDataFrame"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2a2d38 0%, #252833 100%) !important;
        }

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #eef2f7 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="base-input"] *,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #eef2f7 !important;
        }

        [data-testid="stSidebar"] input {
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
            background-color: #121720 !important;
            border-color: rgba(255,255,255,0.28) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
        [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
            color: #f3f7fc !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] * {
            color: #f3f7fc !important;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background: #d49b2d !important;
            box-shadow: 0 0 0 4px rgba(212,155,45,0.22) !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #cdd6e3 !important;
        }

        [data-testid="stSidebar"] button {
            color: #fff6ef !important;
        }

        .hero {
            padding: 1.8rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(30,36,48,0.95), rgba(65,36,21,0.92));
            color: #fff8ef;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 60px rgba(45, 28, 18, 0.18);
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: 0.02em;
        }

        .hero h1,
        .hero h2,
        .hero h3,
        .hero span,
        .hero div {
            color: #fff6ef !important;
        }

        .panel {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 40px rgba(52, 43, 29, 0.08);
        }

        .panel h1,
        .panel h2,
        .panel h3,
        .panel h4,
        .panel p,
        .panel span {
            color: var(--ink) !important;
        }

        .stat-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.4rem;
        }

        .stat-card {
            background: rgba(255,255,255,0.68);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
        }

        .stat-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 700;
        }

        .stat-sub {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.35rem;
        }

        .section-kicker {
            color: var(--accent-dark);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    return requests.get(f"{API_BASE_URL}{path}", timeout=20)


def api_post(path: str, payload=None):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=600)


def request_json(method: str, path: str, payload=None):
    response = api_get(path) if method == "GET" else api_post(path, payload)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=8)
def get_dashboard():
    return request_json("GET", "/dashboard")


@st.cache_data(ttl=8)
def get_history():
    return request_json("GET", "/history")


@st.cache_data(ttl=8)
def get_graph_info():
    return request_json("GET", "/graph-info")


def clear_cache():
    get_dashboard.clear()
    get_history.clear()
    get_graph_info.clear()


def build_comparison_chart(comparison: dict) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=comparison["labels"],
                y=comparison["accuracy"],
                marker=dict(color=["#1f7a76", "#c65f3f"]),
            )
        ]
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="评估状态",
        yaxis_title="准确率",
        font=dict(color="#241b14", size=14),
        xaxis=dict(
            title_font=dict(color="#241b14", size=15),
            tickfont=dict(color="#241b14", size=13),
        ),
        yaxis=dict(
            range=[0, 1],
            title_font=dict(color="#241b14", size=15),
            tickfont=dict(color="#241b14", size=13),
        ),
        legend=dict(font=dict(color="#241b14", size=13)),
    )
    return fig


def build_epoch_curve_chart(latest_attack: dict | None) -> go.Figure:
    fig = go.Figure(
        data=[]
    )
    if latest_attack:
        clean_curve = latest_attack.get("clean_train_curve", {"epochs": [], "values": []})
        poison_curve = latest_attack.get("poison_test_curve", {"epochs": [], "values": []})
        fig.add_trace(
            go.Scatter(
                x=clean_curve["epochs"],
                y=clean_curve["values"],
                mode="lines+markers",
                name="训练集 Train Acc",
                line=dict(color="#1f7a76", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=poison_curve["epochs"],
                y=poison_curve["values"],
                mode="lines+markers",
                name="中毒测试集 Test Acc",
                line=dict(color="#c65f3f", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Epoch",
        yaxis_title="准确率",
        font=dict(color="#241b14", size=14),
        xaxis=dict(
            title_font=dict(color="#241b14", size=15),
            tickfont=dict(color="#241b14", size=13),
        ),
        yaxis=dict(
            range=[0, 1],
            title_font=dict(color="#241b14", size=15),
            tickfont=dict(color="#241b14", size=13),
        ),
        legend=dict(font=dict(color="#241b14", size=13)),
    )
    return fig


try:
    health = api_get("/health")
    health_data = health.json() if health.status_code == 200 else {}
    api_ok = health.status_code == 200 and health_data.get("status") == "ok"
except Exception:
    health_data = {}
    api_ok = False

if "last_attack_result" not in st.session_state:
    st.session_state.last_attack_result = None

st.markdown(
    """
    <div class="hero">
        <div class="section-kicker">Graph Neural Network Poisoning Lab</div>
        <h1>图神经中毒攻击实验平台</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_ok:
    dependency_error = health_data.get("dependency_error")
    if dependency_error:
        st.error(f"后端已启动，但缺少依赖: `{dependency_error}`")
        st.info("请优先在当前项目虚拟环境中补装 `dgl`，然后重新运行 `python backend.py`。")
    else:
        st.error("无法连接后端 API。请先在当前目录运行 `python backend.py`。")
    st.stop()

try:
    dashboard = get_dashboard()
    graph_info = get_graph_info()
except requests.RequestException as exc:
    st.error(f"后端接口请求失败: {exc}")
    st.stop()

metrics = dashboard["metrics"]

with st.sidebar:
    st.markdown("### 实验配置")
    lr = st.number_input("学习率 lr", min_value=0.0001, max_value=1.0, value=0.001, step=0.0005, format="%.4f")
    num_epochs = st.slider("训练轮数 num_epochs", 5, 300, 100, 5)

    st.markdown("### 数据概览")
    st.caption(f"节点数: {graph_info.get('nodes', 0)}")
    st.caption(f"边数: {graph_info.get('edges', 0)}")
    st.caption(f"特征维度: {graph_info.get('features', 0)}")
    st.caption(f"类别数: {graph_info.get('classes', 0)}")
    st.caption(f"图密度: {graph_info.get('density', 0):.6f}")

    run_attack = st.button("执行攻击实验", use_container_width=True, type="primary")
    reset_experiment = st.button("清空历史记录", use_container_width=True)

if run_attack:
    with st.spinner("正在调用后端执行攻击实验..."):
        response = api_post(
            "/attack",
            {
                "n_poison_nodes": min(150, int(graph_info.get("nodes", 2708))),
                "lr": lr,
                "num_epochs": num_epochs,
            },
        )
        if response.status_code == 200:
            st.session_state.last_attack_result = response.json()["data"]
            clear_cache()
            progress = st.progress(0)
            for value in range(100):
                progress.progress(value + 1)
                time.sleep(0.003)
            st.success("攻击实验完成，页面数据已刷新。")
        else:
            st.error(f"执行失败: {response.text}")

if reset_experiment:
    response = api_post("/reset")
    if response.status_code == 200:
        st.session_state.last_attack_result = None
        clear_cache()
        st.success("实验历史已清空。")
    else:
        st.error(f"重置失败: {response.text}")

dashboard = get_dashboard()
history_data = get_history()["items"]
metrics = dashboard["metrics"]
latest_attack = dashboard.get("latest_attack")

st.markdown(
    f"""
        <div class="stat-strip">
        <div class="stat-card">
            <div class="stat-label">中毒测试集 Test Acc</div>
            <div class="stat-value">{metrics['accuracy']:.4f}</div>
            <div class="stat-sub">训练集 Train Acc {metrics['clean_train_accuracy']:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">鲁棒性评分</div>
            <div class="stat-value">{metrics['robustness']:.4f}</div>
            <div class="stat-sub">越接近 1 说明越稳定</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">平均置信度</div>
            <div class="stat-value">{metrics['avg_confidence'] * 100:.1f}%</div>
            <div class="stat-sub">累计实验数 {metrics['experiment_count']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if latest_attack:
    st.info(
        f"最近一次实验: lr {latest_attack['lr']:.4f} | epochs {latest_attack['train_epochs']} | "
        f"train acc {latest_attack['clean_train_accuracy']:.4f} | test acc {latest_attack['poison_accuracy']:.4f} | "
        f"差值 {latest_attack['accuracy_drop']:.4f} | 时间 {latest_attack['timestamp']}"
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Train Acc / Test Acc 对比")
    st.plotly_chart(build_comparison_chart(dashboard["comparison"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 单次训练 Epoch 曲线")
    st.plotly_chart(build_epoch_curve_chart(latest_attack), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([1.3, 1])

with bottom_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 实验历史")
    if history_data:
        history_df = pd.DataFrame(
            [
                {
                    "时间": item["timestamp"],
                    "攻击": item["attack_name"],
                    "lr": item["lr"],
                    "epochs": item["train_epochs"],
                    "Train Acc": item["clean_train_accuracy"],
                    "Poison Test Acc": item["poison_accuracy"],
                    "差值": item["accuracy_drop"],
                    "中毒占比": item["changed_fraction"],
                }
                for item in history_data
            ]
        )
        st.dataframe(history_df, use_container_width=True, height=320)
    else:
        st.write("暂无实验历史，先在左侧运行一次攻击。")
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 最近一次攻击摘要")
    latest = st.session_state.last_attack_result or latest_attack
    if latest:
        st.write("实验方式: `修改节点中毒`")
        st.write(f"学习率 lr: `{latest['lr']:.4f}`")
        st.write(f"训练轮数: `{latest['train_epochs']}`")
        st.write(f"训练集 Train Acc: `{latest['clean_train_accuracy']:.4f}`")
        st.write(f"中毒测试集 Test Acc: `{latest['poison_accuracy']:.4f}`")
        st.write(f"两者差值: `{latest['accuracy_drop']:.4f}`")
        st.write("真实每个 epoch 输出:")
        epoch_lines = []
        history = latest.get("history", [])
        for item in history:
            epoch_label = "Final" if item.get("is_final") else f"Epoch {item['epoch']}"
            eval_text = (
                f"train acc {item['train_acc']:.4f} | poison test acc {item['test_acc']:.4f}"
                if item.get("train_acc") is not None and item.get("test_acc") is not None
                else "train/test 未评估"
            )
            epoch_lines.append(
                f"{epoch_label} | time {item['epoch_time']:.4f}s | {eval_text}"
            )
        st.code("\n".join(epoch_lines) or "暂无 epoch 记录", language="text")
    else:
        st.write("还没有攻击结果。")
    st.markdown("</div>", unsafe_allow_html=True)
