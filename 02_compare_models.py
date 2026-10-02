# -*- coding: utf-8 -*-
"""
02 模型对比：TransE vs DistMult vs ComplEx @ WN18RR
====================================================
做什么：
  用同一套训练设置（同随机种子、同轮数）分别训练三个经典知识图谱嵌入模型，
  在测试集上评估并打印对比表（MRR / Hit@1 / Hit@3 / Hit@10 / Mean Rank）。

怎么跑：python 02_compare_models.py
提示：三个模型依次训练+评估，CPU 上约 30~60 分钟。想快就把 N_EPOCHS 调小。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import torch
from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel, DistMultModel, ComplExModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_wn18rr
from torchkge.evaluation import LinkPredictionEvaluator

EMB_DIM = 100
N_EPOCHS = 100  # 正式对比用 100 轮；只想快速验证流程可临时调小（如 10）
B_SIZE = 1024
LR = 0.0004
MARGIN = 0.5
SEED = 42

use_cuda = 'all' if cuda.is_available() else None

print('加载 WN18RR ……')
kg_train, _, kg_test = load_wn18rr()
print(f'  实体: {kg_train.n_ent}  |  关系: {kg_train.n_rel}  |  训练三元组: {kg_train.n_facts}')

# 三个模型：名称 -> (模型类, 额外构造参数)
MODELS = [
    ('TransE',   TransEModel,   {'dissimilarity_type': 'L2'}),
    ('DistMult', DistMultModel, {}),
    ('ComplEx',  ComplExModel,  {}),
]

results = []
for name, model_cls, kwargs in MODELS:
    print('=' * 60)
    print(f'训练 {name} ……')
    t1 = time.time()

    torch.manual_seed(SEED)                      # 保证三个模型起点一致
    model = model_cls(EMB_DIM, kg_train.n_ent, kg_train.n_rel, **kwargs)
    criterion = MarginLoss(MARGIN)
    if use_cuda:
        model.cuda()
        criterion.cuda()

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=B_SIZE, use_cuda=use_cuda)

    for epoch in range(N_EPOCHS):
        running_loss = 0.0
        for batch in dataloader:
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            optimizer.zero_grad()
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.normalize_parameters()
        print(f'  epoch {epoch + 1}/{N_EPOCHS} | mean loss: {running_loss / len(dataloader):.5f}')

    print(f'评估 {name} ……')
    model.eval()
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(b_size=64)
    _, mr = evaluator.mean_rank()      # 取第二个返回值：过滤口径（与 01 一致）
    _, mrr = evaluator.mrr()
    _, h1 = evaluator.hit_at_k(1)
    _, h3 = evaluator.hit_at_k(3)
    _, h10 = evaluator.hit_at_k(10)
    results.append((name, mrr, h1, h3, h10, mr))
    print(f'  {name}: MRR={mrr:.4f} Hit@1={h1:.4f} Hit@3={h3:.4f} Hit@10={h10:.4f} MeanRank={mr:.1f}  ({(time.time() - t1) / 60:.1f} min)')

print('=' * 60)
print('\n对比结果（过滤口径，越大越好；Mean Rank 越小越好）：\n')
header = f'| 模型 | MRR | Hit@1 | Hit@3 | Hit@10 | Mean Rank |'
sep = '| --- | --- | --- | --- | --- | --- |'
rows = [f'| {n} | {m:.4f} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.1f} |' for n, m, a, b, c, d in results]
table = '\n'.join([header, sep] + rows)
print(table)

with open('results_compare.txt', 'w', encoding='utf-8') as f:
    f.write('知识图谱链接预测模型对比 @ WN18RR（过滤口径）\n')
    f.write(table + '\n')
print('对比表已写入 results_compare.txt。')
print('完成！接下来可以跑 03_predict.py 做预测演示。')
