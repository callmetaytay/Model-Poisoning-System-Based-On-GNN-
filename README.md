# 图神经中毒攻击实验平台

这个仓库现在已经整理成一套可直接演示的 GNN 中毒攻击网站：

- `backend.py` 提供 FastAPI 接口
- `app.py` 提供 Streamlit 可视化网站
- 现有的 `GCN`、`load_data.py`、权重文件和图数据被直接复用

页面支持：

- 查看攻击前后准确率、鲁棒性、检测率
- 查询单节点标签、预测、度数、特征统计
- 查看实验历史记录

## 启动方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
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

如果还没有这个环境，可以这样创建：

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install --upgrade pip setuptools wheel packaging
.venv311/bin/pip install -r requirements.txt
```

注意：

- 不建议使用 Python `3.14` 运行这个项目，DGL 在该版本上容易出现安装和导入问题
- `torchdata` 需要固定为 `0.9.0`，更高版本会移除 DGL 依赖的 `datapipes`
- 本地 `graphgallery` 代码依赖 `yacs`、`scikit-learn`、`tabulate`、`termcolor`、`numba`、`gensim`、`tqdm`，不要漏装

### 2. 启动后端

```bash
python backend.py
```

默认地址：

- 后端 API: `http://127.0.0.1:8000`

### 3. 启动前端网站

```bash
/streamlit run app.py
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
