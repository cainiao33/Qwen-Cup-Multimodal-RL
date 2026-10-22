#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_zip.py —— 只读校验提交 zip（件数/嵌套/CRC/大小/md5）

用法: python verify_zip.py <zip> [期望件数]
只读，不修改任何文件。
"""
import hashlib
import os
import sys
import zipfile

p = sys.argv[1]
exp = int(sys.argv[2]) if len(sys.argv) > 2 else 13
if not os.path.exists(p):
    print("FAIL: 不存在 %s" % p)
    sys.exit(1)
size = os.path.getsize(p)
try:
    z = zipfile.ZipFile(p)
    il = z.infolist()
except Exception as e:
    print("VERDICT: FAIL —— 不是合法 zip（%s），size=%d" % (type(e).__name__, size))
    sys.exit(2)
names = [i.filename for i in il]
nested = [n for n in names if "/" in n or n.endswith("/")]
crc = z.testzip()
h = hashlib.md5()
with open(p, "rb") as f:
    while True:
        b = f.read(1 << 22)
        if not b:
            break
        h.update(b)
print("zip        = %s" % p)
print("size       = %d" % size)
print("md5        = %s" % h.hexdigest())
print("成员数     = %d（期望 %d）" % (len(il), exp))
print("成员       = %s" % sorted(names))
print("嵌套/目录  = %s" % (nested or "无"))
print("CRC 校验   = %s" % (crc or "OK"))
total_u = sum(i.file_size for i in il)
print("解压总大小 = %d" % total_u)
ok = (len(il) == exp and not nested and crc is None)
print("VERDICT: %s" % ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 3)
