# 数据集说明

## 目录结构

请将中草药图像按以下结构组织：

```
中草药datas/
├── Anxixiang/
│   ├── Anxixiang_001.jpg
│   ├── Anxixiang_002.jpg
│   └── ...
├── Baibiandou/
│   ├── Baibiandou_001.jpg
│   └── ...
├── Baifan/
│   └── ...
├── Bailian/
├── Baimaogen/
├── Baiqian/
├── Baishao/
├── Baizhi/
├── Baiziren/
├── Beishashen/
├── Bibo/
├── Bichengqie/
├── Biejia/
├── Binglang/
├── Cangzhu/
├── Caodoukou/
├── Chenxiang/
├── Chuanlianzi/
├── Chuanmuxiang/
├── Chuanniuxi/
├── Dafupi/
├── Dandouchi/
├── Daoya/
├── Dilong/
├── Dongchongxiacao/
├── Fangfeng/
├── Fanxieye/
├── Fengfang/
├── Gancao/
├── Ganjiang/
├── Gansong/
├── Gaoben/
├── Ghishizhi/
├── Gouqizi/
├── Guizhi/
├── Gujingcao/
├── Guya/
├── HaiIong/
├── Haipiaoxiao/
├── Hehuanpi/
├── Huangbo/
├── Huangqi/
├── Huangqin/
├── Hubeibeimu/
├── Jiangcan/
├── Jiezi/
├── Jiguanhua/
├── Jindenglong/
├── Jineijin/
├── Jingjiesui/
├── Jinguolan/
├── Jinqianbaihuashe/
├── Jiuxiangchong/
├── Juhe/
├── Kudiding/
├── Laifuzi/
├── Lianfang/
├── Lianxu/
├── Lianzi/
├── Lianzixin/
├── Lingzhi/
├── Lizhihe/
├── Longyanrou/
├── Lugen/
├── Lulutong/
├── Maidong/
├── Mudingxiang/
├── Qianghuo/
├── Qiannianjian/
├── Qinpi/
├── Quanxie/
└── Rendongteng/
```

## 数据要求

- **格式**：JPG / PNG
- **每类数量**：建议 ≥ 50 张（越多越好）
- **总类别**：72 类
- **命名**：无特殊要求，文件夹名必须与 `config.py` 中的 `actual_classes` 一致

## 数据增强（训练时自动应用）

- 随机水平翻转
- 随机旋转 (±10°)
- 颜色抖动 (亮度/对比度/饱和度 ±0.2)
- 标准化 (ImageNet 均值/方差)

## 数据集划分

代码自动按 **8:2** 划分训练集 / 验证集：
- 训练集：80%
- 验证集：20%

> ⚠️ 本文件仅为说明文档，实际数据集 **不入库**（已在 `.gitignore` 中排除）。
