#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g6d_export_submit_v4.py —— v4 训练完成后：合并 LoRA -> 完整模型，按提交指南整理模型目录
提交指南要求：压缩包根目录直铺 config.json / model.safetensors / generation_config.json
/ preprocessor_config.json / tokenizer_config.json / tokenizer.json 等（不能多一层外层文件夹）
"""
import os, sys, json, shutil
import torch

MODEL = '/data/models/Qwen3.5-0.8B'
ADAPTER = '/data/g6d/output_qwen3p5_v4_lora'
SUBMIT = '/data/g6d/submit_Qwen3p5-0.8B-g6d-v4'


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    os.makedirs(SUBMIT, exist_ok=True)
    print('loading base model for merge (cpu/bf16) ...')
    base = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    print('merging LoRA ...')
    model = model.merge_and_unload()
    print('saving full model ->', SUBMIT)
    model.save_pretrained(SUBMIT)

    must_copy = [
        'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json',
        'video_preprocessor_config.json', 'chat_template.jinja', 'configuration.json',
        'vocab.json', 'merges.txt', 'special_tokens_map.json',
    ]
    for fn in must_copy:
        src = os.path.join(MODEL, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SUBMIT, fn))
        else:
            print('  (skip missing)', fn)

    gen_path = os.path.join(SUBMIT, 'generation_config.json')
    if not os.path.exists(gen_path):
        gen = {
            "bos_token_id": 248045,
            "eos_token_id": 248044,
            "pad_token_id": 248044,
            "do_sample": False,
            "max_new_tokens": 2048,
        }
        with open(gen_path, 'w', encoding='utf-8') as f:
            json.dump(gen, f, ensure_ascii=False, indent=2)
        print('  generation_config.json 已生成')

    print('\n导出目录内容:')
    for fn in sorted(os.listdir(SUBMIT)):
        fp = os.path.join(SUBMIT, fn)
        print(f'  {fn}  {os.path.getsize(fp) if os.path.isfile(fp) else "<dir>"}')


if __name__ == '__main__':
    main()
