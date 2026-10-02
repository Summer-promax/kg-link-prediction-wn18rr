# -*- coding: utf-8 -*-
"""
01 链接预测基线：TransE @ WN18RR
================================
做什么：
  1. 加载 WN18RR 数据集（词汇关系知识图谱，首次运行自动下载）
  2. 训练 TransE 模型（知识图谱嵌入的最经典基线）
  3. 在测试集上评估：MRR / Hit@1 / Hit@3 / Hit@10 / Mean Rank
  4. 画出训练 loss 曲线（loss_transE.png），保存模型（transE_wn18rr.pt）

怎么跑：python 01_transE_baseline.py
想快点验证流程：把下面 N_EPOCHS 改成 2 先跑一遍。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import torch
from torch import cuda
from torch.optim import Adam
from tqdm import tqdm

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_wn18rr
from torchkge.evaluation import LinkPredictionEvaluator

# ---------------- 参数（零基础可以先不改，想提速看注释） ----------------
EMB_DIM = 100     # 向量维度：越大越能记细节，但越慢（每个实体/关系用100个数字描述，维度指向细节数量）
N_EPOCHS = 300    # 训练轮数：官方标准是几百轮，本机约 3 秒/轮，300 轮约 15 分钟
B_SIZE = 1024     # 每批处理的三元组数
LR = 0.0004       # 学习率
MARGIN = 0.5      # 正样本与负样本分数的最小差距
SEED = 42         # 随机种子：固定后每次结果可复现

torch.manual_seed(SEED)
use_cuda = 'all' if cuda.is_available() else None
t0 = time.time()

print('=' * 60)
print('第 1 步 / 3：加载 WN18RR 数据集（首次运行会自动下载，约 1.4 MB）')
kg_train, kg_val, kg_test = load_wn18rr()
print(f'  训练集三元组: {kg_train.n_facts}  |  实体: {kg_train.n_ent}  |  关系: {kg_train.n_rel}')
print(f'  测试集三元组: {kg_test.n_facts}')

print('=' * 60)
print('第 2 步 / 3：训练 TransE')
print(f'  设备: {"CUDA (显卡)" if use_cuda else "CPU"}  |  维度: {EMB_DIM}  |  轮数: {N_EPOCHS}')

model = TransEModel(EMB_DIM, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(MARGIN)
if use_cuda:
    model.cuda()
    criterion.cuda()

optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=B_SIZE, use_cuda=use_cuda)

losses = []
iterator = tqdm(range(N_EPOCHS), unit='epoch', desc='Training')
for epoch in iterator:
    running_loss = 0.0
    for batch in dataloader:
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)     # 随机造一批负样本（错误三元组）
        optimizer.zero_grad()
        pos, neg = model(h, t, r, n_h, n_t)          # 正样本分 vs 负样本分
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    losses.append(epoch_loss)
    iterator.set_description(f'Epoch {epoch + 1} | mean loss: {epoch_loss:.5f}')
    model.normalize_parameters()

print('=' * 60)
print('画训练 loss 曲线（loss_transE.png）……')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(losses, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('TransE training loss (WN18RR)')
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_transE.png', dpi=150, bbox_inches='tight')
    print('  已保存: loss_transE.png（loss 越来越小 = 模型在学）')
except Exception as e:
    print(f'  画图失败（不影响结果，可忽略）: {e}')

print('=' * 60)
print('第 3 步 / 3：测试集评估（要对 4 万多个实体打分，耐心等 2~5 分钟）')
model.eval()
evaluator = LinkPredictionEvaluator(model, kg_test)
evaluator.evaluate(b_size=64)

mr, filt_mr = evaluator.mean_rank()
mrr, filt_mrr = evaluator.mrr()
h1, filt_h1 = evaluator.hit_at_k(1)
h3, filt_h3 = evaluator.hit_at_k(3)
h10, filt_h10 = evaluator.hit_at_k(10)

print('\n结果（过滤后的指标是论文里常用的口径）：')
print(f'  MRR       = {filt_mrr:.4f}')
print(f'  Hit@1     = {filt_h1:.4f}')
print(f'  Hit@3     = {filt_h3:.4f}')
print(f'  Hit@10    = {filt_h10:.4f}')
print(f'  Mean Rank = {filt_mr:.1f}')

with open('results_transE.txt', 'w', encoding='utf-8') as f:
    f.write(f'TransE @ WN18RR (filtered)\nMRR={filt_mrr:.4f} Hit@1={filt_h1:.4f} Hit@3={filt_h3:.4f} Hit@10={filt_h10:.4f} MeanRank={filt_mr:.1f}\n')
print('\n指标已写入 results_transE.txt')

torch.save(model.state_dict(), 'transE_wn18rr.pt')
print('模型已保存: transE_wn18rr.pt（03_predict.py 会用到它）')
print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')
print('完成！接下来可以跑 02_compare_models.py（对比）或 03_predict.py（预测演示）。')
