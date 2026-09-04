# 中草药图像识别系统 (Herbal Identification)

基于 **PyTorch + ResNet18 迁移学习** 的中草药图像分类项目，支持训练、验证、可视化与单图预测全流程。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 项目简介

本项目利用 **迁移学习（Transfer Learning）** 技术，基于预训练的 **ResNet18** 模型，对 **72 种中草药** 图像进行分类识别。通过微调（Fine-tuning）最后几层网络，在保证精度的同时大幅提升训练效率。

### 核心功能

- ✅ **迁移学习**：冻结 ResNet18 浅层，仅微调 Layer4 + FC 层
- ✅ **混合精度训练（FP16）**：加速训练、降低显存占用
- ✅ **梯度检查点（Gradient Checkpointing）**：以时间换空间，支持低显存设备
- ✅ **数据增强**：随机翻转、旋转、颜色抖动
- ✅ **训练可视化**：自动绘制 Loss / Accuracy 曲线
- ✅ **单图预测**：命令行交互式识别中草药

---

## 🗂️ 项目结构

```
herbal_identification/
├── train.py            # 训练主脚本
├── config.py           # 类别配置生成与验证
├── predict.py          # 单张图像预测脚本
├── requirements.txt    # Python 依赖
├── .gitignore
├── README.md
└── output/             # 训练输出（自动生成，不入库）
    ├── best_model.pth      # 最佳模型权重
    ├── metrics.npz         # 训练指标数据
    ├── metrics_plot.png    # 训练曲线图
    ├── class_names.txt     # 类别名称
    └── config.json         # 运行时配置
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 建议使用虚拟环境
conda create -n herbal python=3.9
conda activate herbal

pip install -r requirements.txt
```

### 2. 数据集准备

将中草药图像按 **类别文件夹** 组织：

```
中草药datas/
├── Anxixiang/
│   ├── img1.jpg
│   └── ...
├── Baibiandou/
│   ├── img1.jpg
│   └── ...
├── Baifan/
│   └── ...
└── ...（共 72 个类别）
```

> 每个子文件夹名为一种中草药的英文名称，文件夹内为该草药的图片。

### 3. 生成配置文件

```bash
python config.py
```

输出示例：
```
✅ 配置文件验证通过
总类别数: 72
前5个类别-代码对照：
Anxixiang       -> Axi001
Baibiandou      -> Bai002
Baifan          -> Bai003
Bailian         -> Bai004
Baimaogen       -> Bai005
```

### 4. 开始训练

```bash
python train.py --data_dir "C:\Users\WONDER\Desktop\中草药datas" --epochs 30 --batch_size 8
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | `C:\Users\WONDER\Desktop\中草药datas` | 数据集根目录 |
| `--batch_size` | `8` | 批大小（根据显存调整） |
| `--epochs` | `30` | 训练轮次 |
| `--lr` | `0.001` | 学习率 |
| `--output_dir` | `./output` | 输出目录 |
| `--num_workers` | `2` | 数据加载线程数 |

训练过程中会实时打印：
- GPU 显存使用情况
- 每个 Epoch 的 Train/Val Loss 与 Accuracy
- 自动保存验证集准确率最高的模型

### 5. 单图预测

```bash
python predict.py
```

输出示例：
```
============================================================
中草药智能分类系统
============================================================

🔍 预测结果:
1. 名称: Baiziren (代码: Bai009)
   置信度: 92.35%

2. 名称: Baishao (代码: Bai008)
   置信度: 3.12%
...
```

---

## 🧠 技术原理

### 迁移学习 (Transfer Learning)

利用在 ImageNet 上预训练的 ResNet18 作为 backbone，冻结浅层卷积参数，仅微调 `layer4` 和全连接层：

```python
# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 解冻 layer4 + fc
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
```

**优势：**
- 大幅减少训练参数，加快收敛
- 小数据集也能获得良好效果

### 混合精度训练 (Mixed Precision)

使用 `torch.cuda.amp` 自动混合 FP16/FP32 精度：

```python
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
```

**优势：** 训练速度提升 ~2x，显存占用降低 ~40%。

### 梯度检查点 (Gradient Checkpointing)

将 `layer4` 拆分为 2 段，前向传播时不保存中间激活值，反向传播时重新计算：

```python
checkpoint_sequential(self.layer4, segments=2, input=x)
```

**优势：** 以约 20% 额外计算时间为代价，大幅降低显存峰值。

### 数据增强 (Data Augmentation)

| 变换 | 作用 |
|---|---|
| RandomHorizontalFlip | 水平翻转，增加样本多样性 |
| RandomRotation(10°) | 小幅旋转，增强旋转鲁棒性 |
| ColorJitter | 亮度/对比度/饱和度扰动，增强色彩鲁棒性 |

---

## 📊 训练流程

```
数据集 (72类)
   │
   ▼
数据增强 + 标准化 (224×224)
   │
   ▼
┌─────────────────────────────┐
│   ResNet18 (预训练)          │
│   ├── Conv1 ~ Layer3 (冻结)  │
│   ├── Layer4 (微调 + 检查点) │
│   └── FC (微调 → 72类)       │
└─────────────────────────────┘
   │
   ▼
CrossEntropyLoss + AdamW
   │
   ▼
训练 / 验证 → 保存最佳模型
   │
   ▼
可视化 Loss & Accuracy 曲线
```

---

## 📝 实验结果说明

训练完成后，`output/` 目录包含：

| 文件 | 说明 |
|---|---|
| `best_model.pth` | 验证集准确率最高的模型权重 |
| `metrics.npz` | 每个 Epoch 的 loss / acc 数值 |
| `metrics_plot.png` | 训练/验证 损失与准确率曲线图 |
| `class_names.txt` | 72 个类别名称列表 |
| `config.json` | 类别代码映射配置 |

---

## ⚙️ 系统要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: 可选（CPU 也可训练，速度较慢）
- **显存**: ≥ 4GB（使用梯度检查点后可在 2GB 显存设备运行）
- **数据集**: 72 类中草药图像，建议每类 ≥ 50 张

---

## 📄 License

MIT License
