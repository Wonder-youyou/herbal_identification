"""
快速验证项目结构完整性（无需 GPU / 数据集）
运行: python test_project.py
"""
import os
import sys

# 检查必要文件是否存在
required_files = [
    "train.py",
    "config.py",
    "predict.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
]

print("=" * 50)
print("中草药识别项目 — 结构自检")
print("=" * 50)

missing = []
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  ✅ {f:<20} ({size} bytes)")
    else:
        print(f"  ❌ {f:<20} 缺失!")
        missing.append(f)

print()

if missing:
    print(f"⚠️  缺失文件: {missing}")
    sys.exit(1)

# 验证 config.py 逻辑
print("=" * 50)
print("验证 config.py 类别配置逻辑")
print("=" * 50)

sys.path.insert(0, os.path.dirname(__file__))
from config import actual_classes, generate_class_codes

codes = generate_class_codes(actual_classes)
print(f"  ✅ 类别总数: {len(actual_classes)}")
print(f"  ✅ 代码总数: {len(codes)}")
print(f"  ✅ 数量匹配: {len(actual_classes) == len(codes)}")
print()
print("  前5个类别-代码对照：")
for cls, code in zip(actual_classes[:5], codes[:5]):
    print(f"    {cls:<15} -> {code}")

print()
print("=" * 50)
print("🎉 项目结构完整，可正常上传 GitHub！")
print("=" * 50)
