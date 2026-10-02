# 知识图谱链接预测基线复现：TransE / DistMult / ComplEx @ WN18RR

用 PyTorch + [torchkge](https://github.com/torchkge-team/torchkge) 在 WN18RR 数据集上复现三个经典知识图谱嵌入模型的链接预测基线，并用**同一套超参数、同一随机种子**做受控对比。

- 任务：链接预测（link prediction）——给定 (头实体, 关系)，在全部 40,943 个候选实体中为真实尾实体排序
- 数据：[WN18RR](https://github.com/TimDettmers/ConvE)（WordNet 子集，首次运行自动下载，约 7MB）
- 环境：CPU 即可完整复现（TransE 300 轮约 15 分钟）

## 快速开始

```bash
pip install -r requirements.txt   # torchkge / matplotlib / tqdm
pip install torch                 # torch 单独装，见下方「环境说明」
```

```bash
# 方式一：直接体验（仓库附带训好的权重，无需训练）
python 03_predict.py              # Top-5 预测演示，几秒出结果

# 方式二：完整复现
python 01_transE_baseline.py      # 训练 TransE 300 轮 + 评估（CPU 约 15 分钟）
python 02_compare_models.py       # 三模型受控对比，各 100 轮（CPU 约 30-60 分钟）
```

依赖顺序：`03` 读取 `01` 保存的 `weights/transE_wn18rr.pt`，其余无依赖。`SEED=42` 固定，任何机器上重跑 `01` 应得到一致结果。

## 超参数（要改就改这三个文件顶部的同一组）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `EMB_DIM` | 100 | 实体/关系向量维度 |
| `N_EPOCHS` | 300（01）/ 100（02） | 训练轮数，直接决定运行时长 |
| `B_SIZE` | 1024 | 批大小（WN18RR 训练集 86,845 条，约 85 批/轮） |
| `LR` | 4e-4 | Adam 学习率 |
| `MARGIN` | 0.5 | MarginLoss 间隔（文献常用值） |
| `SEED` | 42 | 随机种子（02 中每个模型训练前都会重置，保证公平对比） |

## 脚本说明

| 脚本 | 做什么 | 产物 |
| --- | --- | --- |
| `01_transE_baseline.py` | 训练 TransE（300 轮）→ 画 loss 曲线 → 测试集评估 | `loss_transE.png`、`results_transE.txt`、`transE_wn18rr.pt` |
| `02_compare_models.py` | 受控对比 TransE / DistMult / ComplEx（各 100 轮），输出 Markdown 对比表 | `results_compare.txt` |
| `03_predict.py` | 加载训好的模型，对真实三元组挖尾猜 Top-5，标注正确答案排名 | 控制台输出（示例见 `results/predict_sample.txt`） |

## 结果

### TransE @ 300 epochs（filtered 口径，seed=42）

| MRR | Hit@1 | Hit@3 | Hit@10 | Mean Rank |
| --- | --- | --- | --- | --- |
| 0.1503 | 0.0029 | 0.2892 | 0.3803 | 7280.0 |

与公开报道的 TransE / WN18RR 基线水平同量级（MRR ≈ 0.15–0.23，取决于维度与轮数配置）。

### 三模型受控对比（各 100 epochs，filtered 口径，seed=42）

控制变量：同数据、同超参、同 seed（且每个模型训练前重置随机序列，保证初始向量和负采样序列一致）——唯一变量是打分函数。

| 模型 | MRR | Hit@1 | Hit@3 | Hit@10 | Mean Rank |
| --- | --- | --- | --- | --- | --- |
| TransE | 0.1463 | 0.0008 | 0.2838 | 0.3681 | 7995.3 |
| DistMult | 0.3607 | 0.3532 | 0.3628 | 0.3743 | 9455.5 |
| ComplEx | 0.4240 | 0.3992 | 0.4373 | 0.4671 | 7566.4 |

与文献中的排序一致：ComplEx > DistMult > TransE——复数向量（ComplEx）能表达不对称关系，点积打分（DistMult）次之，平移打分（TransE）表达力最弱。另一个可观察到的现象：TransE 收敛慢，100 轮（0.1463）与 300 轮（0.1503）差距很小，而其 Hit@1 明显偏低。

## 实验设计说明

- **评估口径**：全部指标为论文标准的 filtered setting——对测试三元组排序时，剔除训练/验证/测试集中出现过的其他正确三元组后再计算排名。唯一例外是 `03` 的演示输出，其排名为未过滤的原始排名。
- **margin = 0.5**：经验超参数，非理论推导；如需严谨可做超参扫描。
- **TransE 的单位球约束**：每轮训练后调用 `normalize_parameters()` 将实体向量归一化，防止模型通过整体放大向量规避 margin 约束。
- **公平性**：`02` 在每个模型训练前重新执行 `torch.manual_seed(SEED)`，重置随机序列，使三个模型的初始权重与负样本完全一致。

## 环境说明与常见问题（Troubleshooting）

实测环境：Windows 10/11，Python 3.11，torch 2.14.0+cpu，torchkge 0.17.7，numpy 2.x（实测兼容）。

1. **`pip install torch` 卡住 / 超时**
   `download.pytorch.org` 直连不稳定。CPU 版直接从 PyPI 装（Windows 上的 PyPI wheel 即 CPU 版），国内可换源：
   ```bash
   pip install torch -i https://mirrors.aliyun.com/pypi/simple/
   ```
2. **镜像 403**：部分镜像对某些请求返回 403，换一家即可（阿里云 / 腾讯云均实测可用）。
3. **需要 GPU 版（CUDA wheel）**：官方 index 超时时，可用镜像站的 `pytorch-wheels/cuXXX` 目录（如上海交大 `mirror.sjtu.edu.cn/pytorch-wheels`、阿里云 `mirrors.aliyun.com/pytorch-wheels`），按显卡代际选 CUDA 标签——RTX 50 系（Blackwell）需 cu128 及以上。本项目 CPU 跑完全够用。
4. **Windows 控制台中文乱码**：脚本已内置 `sys.stdout.reconfigure(encoding="utf-8")`，无需额外处理。
5. **torchkge 版本较老**：实测 0.17.7 与 torch 2.x / numpy 2.x 兼容；若在新版本上报 API 错误，锁回 0.17.7。

## 目录结构

```
├── 01_transE_baseline.py     # 基线训练 + 评估
├── 02_compare_models.py      # 三模型受控对比
├── 03_predict.py             # 预测演示（需 01 的权重）
├── requirements.txt
├── transE_wn18rr.pt          # 预训练好的 TransE 权重（15.6 MB，可直接跑 03）
├── results/
│   ├── results_transE.txt    # 300 轮正式结果
│   ├── results_compare.txt   # 三模型受控对比（100 轮）
│   └── predict_sample.txt    # 03 运行输出示例
└── assets/
    └── loss_transE.png       # loss 收敛曲线
```

## 参考

- Bordes et al. **Translating Embeddings for Modeling Multi-relational Data** (NIPS 2013) — TransE
- Yang et al. **Embedding Entities and Relations for Learning and Inference in Knowledge Bases** (ICLR 2015) — DistMult
- Trouillon et al. **Complex Embeddings for Simple Link Prediction** (ICML 2016) — ComplEx
- Dettmers et al. **Convolutional 2D Knowledge Graph Embeddings** (AAAI 2018) — WN18RR 数据集出处
- [torchkge](https://github.com/torchkge-team/torchkge) — 本项目使用的训练/评估框架
