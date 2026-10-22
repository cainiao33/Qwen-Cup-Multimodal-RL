#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g6d_finetune.py —— 在 Atlas 900 NPU 上对 Qwen3.5-0.8B（多模态）做 LoRA SFT
数据集: g6d/train.jsonl (SWIFT 格式, messages + images)
依赖: transformers 5.15.1 / peft 0.20.0 / torch_npu（通过 PYTHONPATH 隔离目录）

关键经验（已 smoke 验证）:
- 处理器: Qwen3VLProcessor，图片必须嵌进 messages 的 content 结构化列表
  {"type":"image","image":PIL}, {"type":"text","text":...}
- apply_chat_template 返回的 input_ids/attention_mask/mm_token_type_ids 是 1D list
- 模型 forward 要求这些字段为 2D [1, seq]（M-RoPE 会按 batch 索引）
- 必须传入 mm_token_type_ids（来自处理器），否则 M-RoPE 报错
- 多卡 batch: pixel_values 跨样本 concat，image_grid_thw 跨样本 concat
"""
import os, sys, json, argparse, math, signal
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def _alarm_handler(signum, frame):
    raise TimeoutError("sample processing timed out")
signal.signal(signal.SIGALRM, _alarm_handler)

# ---------- 路径 ----------
G6D = '/data/g6d'
MODEL = '/data/models/Qwen3.5-0.8B'
IMG_ROOT = G6D  # train.jsonl 的 images 字段形如 "images/<pool>/xxx.png"，相对 g6d 根目录


def flat_seq(x):
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        return list(x[0])
    return list(x)


def build_messages(q, a, img):
    user = {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": q}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": a}]}
    return user, asst


def process_sample(proc, r, max_len):
    """返回 dict 或 None（跳过）。整体 try/except + 图片尺寸上限，避免异常/卡死样本中断训练。"""
    if not r.get('images'):
        return None
    img_path = r['images'][0]
    if not os.path.isabs(img_path):
        img_path = os.path.join(IMG_ROOT, img_path)
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if min(w, h) < 16 or max(w, h) > 4000:
            return None
        # 限制最长边，避免巨图产生海量 pixel_values 卡死 / OOM
        if max(w, h) > 1280:
            scale = 1280.0 / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        q = r['messages'][0]['content']
        a = r['messages'][1]['content']
        user, asst = build_messages(q, a, img)

        full = proc.apply_chat_template([user, asst], tokenize=True,
                                        add_generation_prompt=False, return_dict=True)
        for k in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
            full[k] = flat_seq(full[k])
        ids = full['input_ids']
        if len(ids) > max_len:
            return None  # 超长直接跳过
        pmsg = proc.apply_chat_template([user], tokenize=True,
                                        add_generation_prompt=True, return_dict=True)
        prompt_ids = flat_seq(pmsg['input_ids'])
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        return {
            'input_ids': ids,
            'attention_mask': full['attention_mask'],
            'mm_token_type_ids': full['mm_token_type_ids'],
            'pixel_values': full['pixel_values'],
            'image_grid_thw': full['image_grid_thw'],
            'labels': labels,
        }
    except Exception:
        return None


class G6DDataset(Dataset):
    def __init__(self, jsonl_path, proc, max_len=2048):
        self.samples = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except Exception:
                    continue
        self.proc = proc
        self.max_len = max_len
        self.proc_ref = proc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        r = self.samples[i]
        # num_workers=0 时主线程可用 signal.alarm 跳过“卡死/异常”样本
        try:
            if hasattr(signal, 'alarm'):
                signal.alarm(30)
            out = process_sample(self.proc_ref, r, self.max_len)
        except Exception:
            out = None
        finally:
            if hasattr(signal, 'alarm'):
                signal.alarm(0)
        if out is None:
            return self._dummy()
        return out

    def _dummy(self):
        return {'_skip': True}


def collate_fn(batch, pad_id):
    batch = [b for b in batch if not b.get('_skip')]
    if len(batch) == 0:
        return None
    seqs = [b['input_ids'] for b in batch]
    max_len = max(len(s) for s in seqs)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    mm = torch.zeros((B, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b['input_ids'])
        input_ids[i, :L] = torch.tensor(b['input_ids'], dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(b['labels'], dtype=torch.long)
        mm[i, :L] = torch.tensor(b['mm_token_type_ids'], dtype=torch.long)
    pv = torch.cat([b['pixel_values'] for b in batch], dim=0)
    grid = torch.cat([b['image_grid_thw'] for b in batch], dim=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'mm_token_type_ids': mm,
        'pixel_values': pv,
        'image_grid_thw': grid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--data', default=os.path.join(G6D, 'train.jsonl'))
    ap.add_argument('--out', default=os.path.join(G6D, 'output_qwen3p5_lora'))
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--logging_steps', type=int, default=20)
    ap.add_argument('--save_steps', type=int, default=2000)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', default='',
                    help='续训: 从该 LoRA checkpoint 目录加载 adapter 后继续训练')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print('device =', device, '| out =', args.out)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    proc = AutoProcessor.from_pretrained(args.model)
    pad_id = proc.tokenizer.pad_token_id
    print('pad_token_id =', pad_id, '| processor:', type(proc).__name__)

    print('loading base model (bf16) ...')
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='eager')
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = get_peft_model(model, lora_config) if not args.resume else PeftModel.from_pretrained(model, args.resume)
    if args.resume:
        print(f'>>> RESUMED from {args.resume}')
    # 确保只有 LoRA 参数可训练（resume 加载后 requires_grad 可能未正确设置）
    for n, p in model.named_parameters():
        p.requires_grad = 'lora' in n.lower()
    model.print_trainable_parameters()
    model.train()

    # Dataset
    ds = G6DDataset(args.data, proc, max_len=args.max_len)
    print('dataset size:', len(ds))
    n_gpu = 1
    total_steps = math.ceil(len(ds) / (args.batch_size * args.grad_accum * n_gpu)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_id),
                        drop_last=False, persistent_workers=False)
    # 优化器 + 调度器
    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    global_step = 0
    best_loss = float('inf')
    running_loss = 0.0
    model.zero_grad()

    print(f'===== START: total_steps≈{total_steps}, warmup={warmup_steps}, eff_bs={args.batch_size*args.grad_accum} =====')
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            with torch.amp.autocast('npu', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    avg = running_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f'[epoch {epoch+1}/{args.epochs}] step {global_step}/{total_steps} '
                          f'loss {avg:.4f} lr {lr:.2e}', flush=True)
                    running_loss = 0.0
                if global_step % args.save_steps == 0 and global_step > 0:
                    ckpt = os.path.join(args.out, f'checkpoint-{global_step}')
                    model.save_pretrained(ckpt)
                    print('saved', ckpt, flush=True)
        # epoch 末保存
        ckpt = os.path.join(args.out, f'checkpoint-epoch{epoch+1}')
        model.save_pretrained(ckpt)
        print('saved', ckpt, flush=True)

    # 最终保存（LoRA adapter）
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    print('===== TRAINING DONE =====')
    print('LoRA adapter saved to', args.out)


if __name__ == '__main__':
    main()
