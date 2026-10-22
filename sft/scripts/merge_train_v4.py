#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_train_v4.py —— 合并 v3 训练数据 + round2 新蒸馏数据 -> train_v4.jsonl
输入:
  g6d/train_v2.jsonl (17924 条, 旧数据)
  data/open_multimodal_v2_round2/reviewed_round2/final_train_round2.jsonl (round2 验真后)
输出:
  g6d/train_v4.jsonl (SWIFT 格式; images 支持相对+绝对路径)
打印: 各池统计 / 总条数
"""
import json, os, sys
from collections import Counter

G6D = '/data/g6d'
R2 = '/data/data/open_multimodal_v2_round2'
OUT = os.path.join(G6D, 'train_v4.jsonl')

# round2 source_dataset -> 六维 pool
POOL_MAP = {
    'GQA': 'spatial', 'PlotQA': 'numeric', 'PhyX': 'physics', 'FSC147': 'counting',
}


def read_jsonl(p):
    rows = []
    if not os.path.exists(p):
        print(f'[WARN] 不存在: {p}')
        return rows
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    old = read_jsonl(os.path.join(G6D, 'train_v2.jsonl'))
    new = read_jsonl(os.path.join(R2, 'reviewed_round2/final_train_round2.jsonl'))
    print(f'旧数据(train_v2): {len(old)} 条')
    print(f'新数据(round2):   {len(new)} 条')

    # round2 记录补齐 pool 字段
    for r in new:
        src = r.get('source', '')
        pool = None
        for k, v in POOL_MAP.items():
            if k.lower() in src.lower():
                pool = v
                break
        r['pool'] = pool or r.get('category', 'visual_perception')

    merged = old + new
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    pools = Counter(r.get('pool', 'unknown') for r in merged)
    print(f'\n合并完成: {len(merged)} 条 -> {OUT}')
    print('按 pool 分布:')
    for k, v in pools.most_common():
        print(f'  {k}: {v}')
    # 图片存在性抽查
    img_ok = 0
    for r in merged[:50] + merged[-50:]:
        p = r.get('images', [''])[0]
        if not p:
            continue
        fp = p if os.path.isabs(p) else os.path.join(G6D, p)
        if os.path.exists(fp):
            img_ok += 1
    print(f'图片抽查(前50+后50): {img_ok}/100 存在')


if __name__ == '__main__':
    main()
