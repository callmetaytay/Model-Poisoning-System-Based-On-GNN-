# 图神经中毒攻击实验平台

这个仓库现在已经整理成一套可直接演示的 GNN 中毒攻击网站：

- `backend.py` 提供 FastAPI 接口
- `app.py` 提供 Streamlit 可视化网站
- 现有的 `GCN`、`load_data.py`、权重文件和图数据被直接复用

页面支持：

- 选择标签中毒 / 特征中毒
- 配置中毒率与中毒节点数
- 查看攻击前后准确率、鲁棒性、检测率
- 查看图结构中毒节点高亮
- 查询单节点标签、预测、度数、特征统计
- 查看实验历史记录

## 启动方式

### 1. 安装依赖

```bash
cd /Users/tay/Desktop/poison
.venv311/bin/pip install -r requirements.txt
```

推荐在 Apple Silicon Mac 上使用 `Python 3.11` 虚拟环境运行本项目。

当前已经验证通过的一组依赖组合是：

- `python 3.11`
- `torch 2.3.0`
- `torchdata 0.9.0`
- `dgl 2.2.1`
- `PyYAML 6.0.2`
- `yacs 0.1.8`
- `scikit-learn 1.5.2`
- `tabulate 0.9.0`
- `termcolor 2.5.0`
- `numba 0.60.0`
- `gensim 4.3.3`
- `tqdm 4.66.5`

如果你还没有这个环境，可以这样创建：

```bash
cd /Users/tay/Desktop/poison
/opt/homebrew/bin/python3.11 -m venv .venv311
.venv311/bin/pip install --upgrade pip setuptools wheel packaging
.venv311/bin/pip install -r requirements.txt
```

注意：

- 不建议使用 Python `3.14` 运行这个项目，DGL 在该版本上容易出现安装和导入问题
- `torchdata` 需要固定为 `0.9.0`，更高版本会移除 DGL 依赖的 `datapipes`
- 本地 `graphgallery` 代码依赖 `yacs`、`scikit-learn`、`tabulate`、`termcolor`、`numba`、`gensim`、`tqdm`，不要漏装

### 2. 启动后端

```bash
cd /Users/tay/Desktop/poison
.venv311/bin/python backend.py
```

默认地址：

- 后端 API: `http://127.0.0.1:8000`

### 3. 启动前端网站

```bash
cd /Users/tay/Desktop/poison
.venv311/bin/streamlit run app.py
```

默认地址：

- 网站主页: `http://localhost:8501`

## API 简表

- `GET /api/health`：服务健康检查
- `GET /api/graph-info`：图基本信息
- `GET /api/graph-nodes`：可视化节点与边
- `GET /api/dashboard`：仪表盘数据
- `GET /api/history`：实验历史
- `GET /api/node-info/{node_id}`：节点详情
- `POST /api/attack`：执行一次中毒实验
- `POST /api/reset`：清空实验历史

## 说明

当前网站更偏“实验展示平台”而不是完整论文复现实验框架：

- 标签中毒会修改节点标签后重新评估
- 特征中毒会向节点特征注入噪声后重新评估

## 环境自检

安装完成后，可以先执行：

```bash
cd /Users/tay/Desktop/poison
.venv311/bin/python - <<'PY'
import torch
import torchdata
import dgl
import yaml
print("torch:", torch.__version__)
print("torchdata:", torchdata.__version__)
print("dgl:", dgl.__version__)
print("yaml ok")
PY
```

预期至少应看到：

- `torch: 2.3.0`
- `torchdata: 0.9.0`
- `dgl: 2.2.1`
- `yaml ok`

如果你下一步想继续，我可以直接帮你再做两种升级：

1. 接入你真实的中毒训练脚本，让网站按钮直接触发真实训练与保存结果
2. 把 Streamlit 改成前后端分离网页，比如 `Vue/React + FastAPI`
