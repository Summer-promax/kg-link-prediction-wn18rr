# -*- coding: utf-8 -*-
"""
03 预测演示：(头实体, 关系) → Top-5 尾实体
============================================
做什么：
  加载 01 训练好的 TransE 模型，取几个真实三元组做演示：
  给出"头实体 + 关系"，让模型预测最可能的 5 个尾实体，
  并告诉你正确答案排在第几名（排名越前说明模型越准）。

怎么跑：python 03_predict.py
前提：先跑过 01_transE_baseline.py，目录里有 transE_wn18rr.pt。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from torchkge.models import TransEModel
from torchkge.utils.datasets import load_wn18rr

EMB_DIM = 100
TOP_K = 5

print('加载数据与模型 ……')
kg_train, _, kg_test = load_wn18rr()
model = TransEModel(EMB_DIM, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
model.load_state_dict(torch.load('transE_wn18rr.pt', map_location='cpu'))
model.eval()

# 实体 id <-> 名字 的对照表
ix2ent = {v: k for k, v in kg_train.ent2ix.items()}
ix2rel = {v: k for k, v in kg_train.rel2ix.items()}


def predict_tails_by_id(h_id, r_id, k=TOP_K):
    """给定头实体 id 和关系 id，返回预测分数最高的 k 个尾实体名字。"""
    t_id = torch.tensor([0], dtype=torch.long)      # 占位，接口需要
    h_t = torch.tensor([h_id], dtype=torch.long)
    r_t = torch.tensor([r_id], dtype=torch.long)
    with torch.no_grad():
        h_emb, _, r_emb, candidates = model.inference_prepare_candidates(
            h_t, t_id, r_t, entities=True)
        scores = model.inference_scoring_function(h_emb, candidates, r_emb)  # (1, n_ent)
        scores = scores[0]
        top = torch.topk(scores, k)
    return top.indices.tolist(), scores


print('\n' + '=' * 66)
print('演示 1：用测试集的前 3 个三元组（挖掉尾实体让模型猜）')
print('=' * 66)
for i in range(3):
    h_id, t_id, r_id = kg_test[i]                  # 一个真实三元组
    h_name, r_name, t_name = ix2ent[h_id], ix2rel[r_id], ix2ent[t_id]
    pred_ids, scores = predict_tails_by_id(h_id, r_id)
    rank = (scores > scores[t_id]).sum().item() + 1   # 正确答案排第几

    print(f'\n题 {i + 1}: ({h_name}, {r_name}, ?)   正确答案: {t_name}   [排第 {rank} 名]')
    for j, pid in enumerate(pred_ids, 1):
        mark = '  ← 正确答案' if pid == t_id else ''
        print(f'  Top{j}: {ix2ent[pid]}{mark}')


print('\n' + '=' * 66)
print('演示 2：从训练集再取 2 个三元组试试')
print('=' * 66)
for i in range(2):
    h_id, t_id, r_id = kg_train[i]
    h_name, r_name, t_name = ix2ent[h_id], ix2rel[r_id], ix2ent[t_id]
    pred_ids, scores = predict_tails_by_id(h_id, r_id)
    rank = (scores > scores[t_id]).sum().item() + 1
    print(f'\n({h_name}, {r_name}, ?)   正确答案: {t_name}   [排第 {rank} 名]')
    for j, pid in enumerate(pred_ids, 1):
        mark = '  ← 正确答案' if pid == t_id else ''
        print(f'  Top{j}: {ix2ent[pid]}{mark}')

print('\n' + '=' * 66)
print('解释：WN18RR 考的是"词和词的关系"，这里的实体是 WordNet 的')
print('8 位数字词义 ID（如 6845599 代表一个词义）。模型没背答案，')
print('是靠训练中学到的向量规律猜出来的。排名越靠前越准。')
