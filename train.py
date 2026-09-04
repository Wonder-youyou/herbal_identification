import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，用于服务器环境
import matplotlib.pyplot as plt
import os
import time
import argparse
import numpy as np
import threading

# 固定随机种子，确保实验结果可复现
torch.manual_seed(42)
np.random.seed(42)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='中草药图像分类训练脚本（优化版）')
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\WONDER\Desktop\中草药datas",
                        help='数据集根目录路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小（根据GPU内存调整）')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出结果保存目录')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载进程数')
    return parser.parse_args()


def load_data(data_dir, batch_size, num_workers):
    """加载数据集并划分训练集和验证集"""
    # 定义数据预处理流程，包括数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 加载完整数据集
    full_dataset = ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes  # 获取类别名称

    # 按8:2比例划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # 取消内存锁定，避免GPU内存问题
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )

    # 打印数据集信息
    print(f"数据集加载完成 - 类别数: {len(class_names)}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    return train_loader, val_loader, class_names


def create_model(num_classes):
    """创建基于ResNet18的分类模型，应用梯度检查点"""
    # 加载预训练的ResNet18模型
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 冻结所有参数，不进行更新
    for param in model.parameters():
        param.requires_grad = False

    # 解冻layer4和全连接层参数，进行微调
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # 修改最后一层全连接层，适应分类类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 定义梯度检查点子模块
    class CheckpointedLayer4(nn.Module):
        def __init__(self, layer4):
            super().__init__()
            self.layer4 = layer4

        def forward(self, x):
            # 将layer4分成2段进行梯度检查点计算，减少内存占用
            return checkpoint_sequential(self.layer4, 2, x)

    # 替换原始layer4为检查点版本
    model.layer4 = CheckpointedLayer4(model.layer4)

    return model


def monitor_memory():
    """监控GPU内存使用情况"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # 总内存(GB)
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3  # 已分配内存(GB)
        cached = torch.cuda.memory_reserved(0) / 1024 ** 3  # 缓存内存(GB)
        usage = allocated / total  # 内存使用率
        return usage, total, allocated, cached
    return 0, 0, 0, 0  # CPU环境返回0


def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, output_dir):
    """训练模型主函数，包含训练和验证过程"""
    train_losses = []  # 保存训练损失
    val_losses = []    # 保存验证损失
    train_acc = []     # 保存训练准确率
    val_acc = []       # 保存验证准确率
    best_val_acc = 0.0  # 保存最佳验证准确率
    scaler = GradScaler()  # 初始化混合精度训练缩放器

    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录 epoch 开始时间
        model.train()  # 设置模型为训练模式
        train_loss = 0.0  # 初始化训练损失
        train_correct = 0  # 初始化训练正确数
        train_total = 0    # 初始化训练样本总数

        # 监控并打印GPU内存使用情况
        mem_usage, total, allocated, cached = monitor_memory()
        print(
            f"Epoch {epoch + 1}/{num_epochs} - GPU内存: 总量 {total:.2f}GB, 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB, 使用率 {mem_usage * 100:.2f}%")

        # 遍历训练数据
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移至设备

            # 混合精度训练上下文管理器
            with autocast():
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

            # 反向传播和优化
            optimizer.zero_grad()  # 梯度清零
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器

            # 统计训练指标
            train_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测类别
            train_total += labels.size(0)  # 累加样本数
            train_correct += predicted.eq(labels).sum().item()  # 累加正确数

            # 定期清理GPU缓存，释放内存
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                mem_usage, _, allocated, cached = monitor_memory()
                print(
                    f"Epoch {epoch + 1} - Batch {batch_idx + 1}/{len(train_loader)} - 清理缓存后: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

        # 计算 epoch 训练指标
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0  # 初始化验证损失
        val_correct = 0  # 初始化验证正确数
        val_total = 0    # 初始化验证样本总数

        # 不计算梯度，节省内存和计算资源
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 计算 epoch 验证指标
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        epoch_time = time.time() - start_time  # 计算 epoch 耗时
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% - "
              f"Time: {epoch_time:.2f}s")

        # 保存最佳模型（基于验证集准确率）
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证集准确率: {best_val_acc:.2f}%")

    # 保存训练指标到文件
    metrics_path = os.path.join(output_dir, 'metrics.npz')
    np.savez(
        metrics_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_acc=np.array(train_acc),
        val_acc=np.array(val_acc)
    )
    print(f"训练指标已保存至: {metrics_path}")

    return train_losses, val_losses, train_acc, val_acc


def plot_metrics(train_losses, val_losses, train_acc, val_acc, output_dir):
    """绘制训练过程中的损失和准确率曲线"""
    plt.figure(figsize=(15, 5))  # 创建图表

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue', linewidth=2)
    plt.plot(val_losses, label='验证损失', color='red', linewidth=2)
    plt.title('训练与验证损失曲线', fontsize=14)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
    plt.legend(fontsize=12)  # 添加图例
    plt.ylim(bottom=0)  # 设置y轴下限为0
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='训练准确率', color='blue', linewidth=2)
    plt.plot(val_acc, label='验证准确率', color='red', linewidth=2)
    plt.title('训练与验证准确率曲线', fontsize=14)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)  # 设置y轴范围为0-100
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()  # 自动调整布局
    plot_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # 保存图表
    print(f"图表已保存至: {plot_path}")
    plt.close()  # 关闭图表，释放资源


def generate_plot_async(train_losses, val_losses, train_acc, val_acc, output_dir):
    """异步生成图表，避免阻塞主线程"""
    thread = threading.Thread(
        target=plot_metrics,
        args=(train_losses, val_losses, train_acc, val_acc, output_dir)
    )
    thread.daemon = True  # 设置为守护线程
    thread.start()  # 启动线程


def save_class_names(class_names, output_dir):
    """保存类别名称到文件"""
    class_names_path = os.path.join(output_dir, 'class_names.txt')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')  # 每个类别名称占一行
    print(f"类别名称已保存至: {class_names_path}")


def main():
    """主函数，程序入口"""
    args = parse_args()  # 解析命令行参数

    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 '{args.data_dir}' 不存在")
        return

    # 选择计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, val_loader, class_names = load_data(
        args.data_dir,
        args.batch_size,
        args.num_workers
    )

    # 创建模型
    model = create_model(len(class_names))
    model = model.to(device)  # 将模型移至指定设备

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)  # AdamW优化器

    # 开始训练模型
    print("开始训练...")
    train_losses, val_losses, train_acc, val_acc = train_model(
        model, criterion, optimizer, train_loader, val_loader, device,
        args.epochs, args.output_dir
    )

    # 异步生成图表
    generate_plot_async(train_losses, val_losses, train_acc, val_acc, args.output_dir)

    # 保存类别名称
    save_class_names(class_names, args.output_dir)

    # 释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_usage, _, allocated, _ = monitor_memory()
        print(f"训练完成 - 最终GPU内存使用: {allocated:.2f}GB, 使用率: {mem_usage * 100:.2f}%")

    print("所有任务已完成！")


if __name__ == "__main__":
    main()  # 调用主函数
