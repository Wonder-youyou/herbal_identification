import torch
import torch.nn as nn
from PIL import Image
import json
import os
import logging
from colorama import Fore, Style, init
from typing import List, Dict
import torchvision.models as models
from torchvision import transforms

# 初始化colorama，用于命令行颜色输出
init(autoreset=True)
# 配置日志记录，同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)


class HerbalClassifier(nn.Module):
    """中草药分类模型，基于ResNet18架构，匹配训练时的模型结构"""

    def __init__(self, num_classes: int):
        super().__init__()
        # 加载预训练的ResNet18模型，不使用预训练权重
        self.backbone = models.resnet18(weights=None)

        # 定义梯度检查点子模块，用于减少训练时的内存占用
        class CheckpointedLayer4(nn.Module):
            def __init__(self, layer4):
                super().__init__()
                self.layer4 = layer4

            def forward(self, x):
                # 导入梯度检查点函数，将layer4分成2段进行计算
                from torch.utils.checkpoint import checkpoint_sequential
                return checkpoint_sequential(self.layer4, segments=2, input=x, use_reentrant=False)

        # 替换原始layer4为检查点版本
        self.backbone.layer4 = CheckpointedLayer4(self.backbone.layer4)
        # 修改最后一层全连接层，适应分类类别数
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        # 前向传播，直接使用backbone处理输入
        return self.backbone(x)


class HerbalPredictor:
    def __init__(self, model_path: str = "best_model.pth", config_path: str = "config.json"):
        """中草药预测器初始化，加载模型和配置"""
        # 规范化模型和配置文件路径为绝对路径
        self.model_path = os.path.abspath(model_path)
        self.config_path = os.path.abspath(config_path)
        # 自动检测并设置设备（GPU优先）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")

        # 加载并验证配置文件
        self.config = self._load_config()
        # 初始化分类模型
        self.model = HerbalClassifier(len(self.config['classes'])).to(self.device)
        # 加载模型权重
        self._load_model()
        # 设置模型为评估模式
        self.model.eval()
        # 初始化图像预处理管道
        self.transform = self._init_transforms()

    def _load_config(self):
        """加载并验证配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证配置文件包含必要字段
        required = ['classes', 'class_codes', 'img_size']
        for field in required:
            if field not in config:
                raise ValueError(f"配置缺少字段: {field}")
        if len(config['classes']) != len(config['class_codes']):
            raise ValueError("类别与代码数量不匹配")
        return config

    def _load_model(self):
        """加载模型权重并处理键名不匹配问题"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")

        try:
            # 加载模型权重，指定映射到当前设备
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

            # 处理键名不匹配问题，主要是处理layer4的双重前缀问题
            new_checkpoint = {}
            for key, value in checkpoint.items():
                if key.startswith('backbone.layer4.layer4'):
                    new_key = key.replace('backbone.layer4.layer4', 'backbone.layer4')
                elif not key.startswith('backbone.'):
                    new_key = f'backbone.{key}'
                else:
                    new_key = key
                new_checkpoint[new_key] = value

            # 加载模型权重，宽松模式允许部分键不匹配
            missing, unexpected = self.model.load_state_dict(new_checkpoint, strict=False)
            if missing:
                logging.warning(f"缺失键: {missing}")
            if unexpected:
                logging.warning(f"多余键: {unexpected}")
            logging.info("模型加载成功")

        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _init_transforms(self):
        """初始化与训练一致的图像预处理流程"""
        return transforms.Compose([
            # 调整图像大小为模型输入要求的尺寸
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            # 随机水平翻转，增强模型泛化能力
            transforms.RandomHorizontalFlip(),
            # 转换为张量
            transforms.ToTensor(),
            # 图像标准化，使用ImageNet的均值和标准差
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path: str, topk: int = 5) -> List[Dict]:
        """对输入图像进行预测，返回前topk个预测结果"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")

        try:
            # 加载图像并转换为RGB格式
            image = Image.open(image_path).convert('RGB')
            # 应用预处理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 禁用梯度计算，提高推理效率
            with torch.no_grad():
                # 模型前向传播
                outputs = self.model(input_tensor)
                # 计算类别概率
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                # 获取topk个最高概率的类别
                topk_values, topk_indices = torch.topk(probs, topk)

            # 构建预测结果
            results = []
            for i in range(topk):
                class_idx = topk_indices[i].item()
                results.append({
                    "class": self.config['classes'][class_idx],
                    "code": self.config['class_codes'][class_idx],
                    "confidence": float(topk_values[i].item()) * 100
                })

            # 过滤低置信度结果
            threshold = self.config.get('confidence_threshold', 0.1)
            valid_results = [r for r in results if r["confidence"] >= threshold * 100]
            return valid_results if valid_results else [{"class": "未识别", "code": "UNK", "confidence": 0}]

        except Exception as e:
            logging.error(f"预测异常: {str(e)}", exc_info=True)
            return [{"error": f"预测失败: {str(e)}"}]


# 主程序入口
if __name__ == "__main__":
    try:
        # 初始化预测器，指定模型和配置文件路径
        predictor = HerbalPredictor(
            model_path=r"./output/best_model.pth",
            config_path=r"./output/config.json"
        )

        # 指定测试图像路径
        test_image = r"C:\Users\WONDER\Desktop\中草药datas\Baiziren\BaizirenBaizirenBaiziren2.jpg"
        if not os.path.exists(test_image):
            raise FileNotFoundError(f"测试图像不存在: {test_image}")

        # 执行预测
        results = predictor.predict(test_image)

        # 打印预测结果
        print("\n" + "=" * 60)
        print(f"{Fore.GREEN}中草药智能分类系统{Style.RESET_ALL}")
        print("=" * 60 + "\n")

        if "error" in results[0]:
            print(f"{Fore.RED}❌ {results[0]['error']}{Style.RESET_ALL}")
        else:
            print("🔍 预测结果:")
            for i, res in enumerate(results):
                print(f"{i + 1}. 名称: {res['class']} (代码: {res['code']})")
                print(f"   置信度: {Fore.YELLOW}{res['confidence']:.2f}%{Style.RESET_ALL}\n")

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"{Fore.RED}❌ 程序异常: {str(e)}{Style.RESET_ALL}")
        logging.error(f"主程序错误: {str(e)}", exc_info=True)
