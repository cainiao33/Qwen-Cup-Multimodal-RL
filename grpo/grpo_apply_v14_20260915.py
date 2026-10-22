#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grpo_apply_v14.py —— 贴牌 RL（复刻 rl4 配方, use_vllm=False）

历史: rl4 配方(9/9 实证 +1.0 无损): abstract rl3 1709 条 / 10 steps / lr 1e-6 / beta 0.2
  / num_gen 8 / 纯 acc boxed reward(对1.0 格式对错0.05 无0) / 权重几乎不动

[9/13 补丁 c · 奖励口径对齐平台]
  · 提取器由 re.search(r"\\boxed\\{([^}]+)\\}") 换成平台官方口径
    rfind("\\boxed{"|"boxed{") + 括号深度计数（与 eval_val.py 逐字一致）。

[9/14 补丁 d · --seed]
  · 新增 --seed（默认 42），同配方换种子 ⇒ 多 run 采样取最高分。

[9/15 补丁 e · P0-1 复合奖励 + 口径对齐 + 按维度诊断]
  1) **norm 对齐 eval_val.py**：新增 norm_eval（去 \\s$\\,，。:：;；、（）()）+ lower），
     并作为默认；旧的去 {} 版本保留为 --norm legacy。此前两者在 `：；、（）()` 与 `,` 上不同。
  2) --reward 增加 **composite** 分支（默认参数下与 platform 等价，保证安全）：
       · --part_credit  语义等价但字面不同的部分分（默认 0.0 = 严格平台口径）
       · --len_penalty / --len_target  completion 过长按比例扣分（默认 0 = 关）
     计分骨架：抽不到=0.0 / 归一相等=1.0 / 语义等价=part_credit / boxed 但错=0.05。
     ⚠️ part_credit>0 会引入"猜一个是/否"的退化激励，务必先小值试（建议 ≤0.2）。
  3) **答案类型感知**：bool / choice / num / text 分类 + 按类型分解的「参考答案自洽率」。
     --dry_load 会打印分组自洽率；任一类型 < 100% 说明该类型在奖励下拿不到 1.0 ⇒ 白训。
  4) **按维度诊断**：数据带 pool 列时，训练期按 pool 累计平均奖励并在每 20 次奖励调用打印。

CLI:
  python grpo_apply_v14.py --model <submit_dir> --data <jsonl> --out <dir>
      [--steps 30] [--lr 1e-6] [--beta 0.2] [--device npu:0]
      [--reward platform|legacy|composite] [--norm eval|legacy]
      [--part_credit 0.0] [--len_penalty 0.0] [--len_target 200]
      [--seed 42] [--save_steps 10] [--save_total_limit 2] [--save_only_model]
      [--dry_load]
"""
import json, os, re, sys, argparse

# 注意：transformers / trl / PIL 一律惰性导入。
# 这样 --dry_load 可以只用标准库跑（不占 NPU），当零成本预筛用。

REWARD_MODE = "platform"
NORM_MODE = "eval"
PART_CREDIT = 0.0
LEN_PENALTY = 0.0
LEN_TARGET = 200


def _norm_eval(s):
    """与 eval_val.py 的 norm() 逐字一致（平台同款口径）。"""
    return re.sub(r"[\s$\\,，。:：;；、（）()]", "", str(s)).strip().lower()


def _norm_legacy(s):
    """9/13~9/14 用的口径（多剥 {} 与英文逗号，少剥 ：；、（））。"""
    return re.sub(r"[\s$\\{},.，。]", "", str(s)).strip().lower()


def _norm(s):
    return _norm_eval(s) if NORM_MODE == "eval" else _norm_legacy(s)


def _decode(ids):
    try:
        return " ".join(str(t) for t in ids)
    except Exception:
        return str(ids)


def extract_platform(text):
    """平台官方口径：与 eval_val.py 的 extract_boxed 逐字一致（rfind + 括号深度计数）。"""
    if not text:
        return None
    last_idx, matched = -1, ""
    for p in ("\\boxed{", "boxed{"):
        idx = text.rfind(p)
        if idx > last_idx:
            last_idx, matched = idx, p
    if last_idx == -1:
        return None
    start = last_idx + len(matched)
    depth = 1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_legacy(text):
    """9/13 之前的历史口径（仅用于复现旧 run）。"""
    m = re.search(r"\\boxed\{([^}]+)\}", text or "", re.DOTALL)
    return m.group(1).strip() if m else None


def _extract(text):
    if REWARD_MODE == "legacy":
        return extract_legacy(text)
    return extract_platform(text)


# ---------------- P0-1: 答案类型感知 ----------------
BOOL_T = {"是", "对", "正确", "能", "可以", "有", "真", "yes", "true", "y", "t", "√"}
BOOL_F = {"否", "不", "不是", "错", "错误", "不能", "不可以", "没有", "假",
          "no", "false", "n", "f", "×", "x"}
_CHOICE_RE = re.compile(r"^[（(\[]?\s*([A-Da-d])\s*[）)\].、,，:：]?\s*$")


def _as_num(t):
    t = str(t).strip().replace("$", "").replace("\\", "").replace(",", "").replace("，", "")
    try:
        return float(t)
    except Exception:
        return None


def answer_type(s):
    t = str(s).strip()
    tl = t.lower()
    if tl in BOOL_T or tl in BOOL_F:
        return "bool"
    if _CHOICE_RE.match(t):
        return "choice"
    if _as_num(t) is not None:
        return "num"
    return "text"


def sem_eq(pred, gt):
    """语义等价的宽松判定；仅在 part_credit > 0 时被调用（最多给 part_credit 分）。"""
    tp, tg = answer_type(pred), answer_type(gt)
    if tp != tg:
        return False
    ps, gs = str(pred).strip().lower(), str(gt).strip().lower()
    if tg == "bool":
        return (ps in BOOL_T and gs in BOOL_T) or (ps in BOOL_F and gs in BOOL_F)
    if tg == "choice":
        mp, mg = _CHOICE_RE.match(str(pred).strip()), _CHOICE_RE.match(str(gt).strip())
        return bool(mp and mg and mp.group(1).lower() == mg.group(1).lower())
    if tg == "num":
        a, b = _as_num(pred), _as_num(gt)
        return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return False


def _score_one(gen, gt):
    """单条打分。默认(part_credit=0,len_penalty=0) 下与平台口径完全一致。"""
    if not isinstance(gen, str):
        gen = _decode(gen) if (isinstance(gen, list) and gen and isinstance(gen[0], int)) else str(gen)
    pred = _extract(gen)
    if pred is None:
        s = 0.0
    elif _norm(pred) == _norm(gt):
        s = 1.0
    elif PART_CREDIT > 0 and sem_eq(pred, gt):
        s = PART_CREDIT
    else:
        s = 0.05
    if LEN_PENALTY > 0 and LEN_TARGET > 0 and len(gen) > LEN_TARGET:
        over = (len(gen) - LEN_TARGET) / float(LEN_TARGET)
        s = max(0.0, s - LEN_PENALTY * min(1.0, over))
    return s


_DIM_STATS = {}
_CALLS = [0]


def reward_boxed(completions, ground_truth, pool=None, **kw):
    """主奖励。pool 列可选，仅用于按六维诊断（GRPO 组内标准化 ⇒ 权重是常量，不影响梯度）。"""
    scores = []
    for i, (gen, gt) in enumerate(zip(completions, ground_truth)):
        s = _score_one(gen, gt)
        scores.append(s)
        if pool is not None:
            d = pool[i] if i < len(pool) else "?"
            st = _DIM_STATS.setdefault(str(d), [0, 0.0])
            st[0] += 1
            st[1] += s
    _CALLS[0] += 1
    if pool is not None and _CALLS[0] % 5 == 0:
        # _DIM_STATS 是自训练开始以来的累积量 ⇒ 打印的是「累计平均奖励」
        parts = ["%s=%.3f(n=%d)" % (k, v[1] / max(v[0], 1), v[0])
                 for k, v in sorted(_DIM_STATS.items())]
        print("[dim-reward-cum @step~%d] %s" % (_CALLS[0], "  ".join(parts)), flush=True)
    return scores


def txt_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            it.get("text", "") for it in content
            if isinstance(it, dict) and it.get("type") == "text")
    return ""


def load_data(path, fast=False):
    """加载 RL 数据。返回 (data, stats)。

    fast=True 不打开图片（--dry_load 用）。
    stats 里带「按答案类型分解的参考答案自洽率」——任一类型 <100% 说明该类型白训。
    """
    data = []
    n_line = n_noimg = n_noans = n_nogt = 0
    ref_hit = ref_miss = 0
    hit_by_type, miss_by_type = {}, {}
    miss_samples = []
    type_dist, pool_dist = {}, {}
    if not os.path.exists(path):
        print("数据不存在: %s" % path, flush=True)
        return data, {"error": "missing"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_line += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            img = (r.get("images") or [None])[0]
            if not img or not os.path.exists(img):
                n_noimg += 1
                continue
            ms = r.get("messages") or []
            if len(ms) < 2:
                n_noans += 1
                continue
            u = txt_of(ms[0].get("content"))
            ans = txt_of(ms[1].get("content"))
            q = u.replace("<image>", "").strip()
            gt = _extract(ans) or r.get("ground_truth")
            if not gt or not q:
                n_nogt += 1
                continue
            gt = str(gt).strip()
            at = answer_type(gt)
            type_dist[at] = type_dist.get(at, 0) + 1
            pd = str(r.get("pool", "?"))
            pool_dist[pd] = pool_dist.get(pd, 0) + 1
            pred = _extract(ans)
            if pred is not None and _norm(pred) == _norm(gt):
                ref_hit += 1
                hit_by_type[at] = hit_by_type.get(at, 0) + 1
            else:
                ref_miss += 1
                miss_by_type[at] = miss_by_type.get(at, 0) + 1
                if len(miss_samples) < 3:
                    miss_samples.append((r.get("id"), at, ans[-70:], gt))
            if fast:
                data.append({"prompt_q": q, "image": img, "ground_truth": gt})
            else:
                from PIL import Image
                data.append({
                    "prompt": [{"role": "user", "content": [
                        {"type": "image", "image": Image.open(img).convert("RGB")},
                        {"type": "text", "text": q},
                    ]}],
                    "ground_truth": gt,
                    "pool": pd,
                })
    stats = {
        "lines": n_line, "kept": len(data), "no_img": n_noimg,
        "no_ans": n_noans, "no_gt": n_nogt,
        "ref_hit": ref_hit, "ref_miss": ref_miss, "miss_samples": miss_samples,
        "hit_by_type": hit_by_type, "miss_by_type": miss_by_type,
        "type_dist": type_dist, "pool_dist": pool_dist,
    }
    print("RL 数据: %d 条 (%s)" % (len(data), path), flush=True)
    return data, stats


def main():
    global REWARD_MODE, NORM_MODE, PART_CREDIT, LEN_PENALTY, LEN_TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=False)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--reward", choices=["platform", "legacy", "composite"], default="platform")
    ap.add_argument("--norm", choices=["eval", "legacy"], default="eval",
                    help="比对口径。eval = 与 eval_val.py 逐字一致（默认，推荐）；"
                         "legacy = 9/13~9/14 旧口径，仅用于复现旧 run。")
    ap.add_argument("--part_credit", type=float, default=0.0,
                    help="语义等价但字面不同的部分分（0=严格平台口径）。>0 有退化激励风险，建议 ≤0.2。")
    ap.add_argument("--len_penalty", type=float, default=0.0,
                    help="completion 超长扣分系数（0=关）。扣分在最终分上做，最多扣满该系数。")
    ap.add_argument("--len_target", type=int, default=200,
                    help="超长惩罚的起始字符数（--len_penalty>0 时生效）。")
    ap.add_argument("--dry_load", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="训练/采样随机种子。默认 42（与历史 rl4/rlx30/rly 同口径）；"
                         "改此值即可产出同配方的『新 run 样本』，用于多 run 采样取最高分。")
    ap.add_argument("--save_steps", type=int, default=5)
    ap.add_argument("--save_total_limit", type=int, default=0)
    ap.add_argument("--save_only_model", action="store_true")
    a = ap.parse_args()

    REWARD_MODE = a.reward
    NORM_MODE = a.norm
    PART_CREDIT = a.part_credit
    LEN_PENALTY = a.len_penalty
    LEN_TARGET = a.len_target

    if a.dry_load:
        data, st = load_data(a.data, fast=True)
        print("=" * 78)
        print("reward 口径 = %s | norm = %s | part_credit = %.2f | len_penalty = %.2f@%d"
              % (REWARD_MODE, NORM_MODE, PART_CREDIT, LEN_PENALTY, LEN_TARGET))
        print("  源行数            = %d" % st.get("lines", 0))
        print("  可进池            = %d" % st.get("kept", 0))
        print("  无图 / 无答案 / 无 boxed = %d / %d / %d"
              % (st.get("no_img", 0), st.get("no_ans", 0), st.get("no_gt", 0)))
        hit, miss = st.get("ref_hit", 0), st.get("ref_miss", 0)
        tot = hit + miss
        print("  ★总参考答案自洽率 = %d/%d (%.2f%%)"
              % (hit, tot, 100.0 * hit / max(tot, 1)))
        hb = st.get("hit_by_type", {})
        mb = st.get("miss_by_type", {})
        bad = []
        for at in sorted(set(list(hb) + list(mb))):
            h, m = hb.get(at, 0), mb.get(at, 0)
            print("    · %-7s 自洽 %d/%d (%.2f%%)" % (at, h, h + m, 100.0 * h / max(h + m, 1)))
            if m > 0:
                bad.append(at)
        print("  答案类型分布: %s" % dict(sorted(st.get("type_dist", {}).items(),
                                                key=lambda kv: -kv[1])))
        print("  维度分布    : %s" % dict(sorted(st.get("pool_dist", {}).items(),
                                                key=lambda kv: -kv[1])))
        for i, at, tail, gt in st.get("miss_samples", []):
            print("    自洽失败样例 [%s/%s] gt=%r ...%r" % (i, at, gt, tail))
        print("VERDICT: %s" % ("OK" if (tot > 0 and miss == 0) else "WARN(白训风险:%s)" % bad))
        sys.exit(0)

    if not a.model or not a.out:
        print("ERROR: 训练模式必须给 --model 与 --out")
        sys.exit(2)

    data, st = load_data(a.data)
    if len(data) < 50:
        print("数据不足 (当前 %d 条)" % len(data), flush=True)
        sys.exit(1)
    tot = st["ref_hit"] + st["ref_miss"]
    if tot and st["ref_miss"] > 0:
        print("[WARN] 参考答案自洽率 %d/%d，有 %d 条参考答案在奖励下拿不到 1.0（按类型: %s）"
              % (st["ref_hit"], tot, st["ref_miss"], st.get("miss_by_type")), flush=True)

    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    proc = AutoProcessor.from_pretrained(a.model)
    args = GRPOConfig(
        output_dir=a.out,
        use_vllm=False,
        num_generations=8,
        max_completion_length=256,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=a.steps,
        learning_rate=a.lr,
        beta=a.beta,
        logging_steps=1,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=(a.save_total_limit or None),
        save_only_model=a.save_only_model,
        seed=a.seed,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=[reward_boxed],
        args=args,
        train_dataset=data,
        processing_class=proc,
    )
    trainer.train()
    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()
