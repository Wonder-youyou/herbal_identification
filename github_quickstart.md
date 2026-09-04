# 中草药识别项目 — GitHub 上传完整指南

## 前置条件
- 已安装 Git
- 已注册 GitHub 账号（假设用户名为 `wonder-you-you`）
- 项目文件夹路径：`C:\Users\WONDER\...`（根据实际情况调整）

---

## 方式一：使用 GitHub Desktop（推荐，图形化）

### 1. 打开 GitHub Desktop
- 点 "Add an Existing Repository from your Hard Drive"
- 选择 `herbal_identification` 文件夹

### 2. 创建仓库
- 若提示 "Not a Git repository" → 点 "Create a Repository"
- Name: `herbal-identification`
- Description: `基于 PyTorch + ResNet18 迁移学习的中草药图像分类系统`
- 选 **Public**
- ✅ 勾选 "Initialize with README" → **不勾选**（已有 README）

### 3. 提交 + 发布
- Summary: `feat: 初始化中草药识别项目（训练/预测/配置）`
- 点 **Commit to main**
- 点 **Publish repository** → 等待上传完成

---

## 方式二：使用命令行

```bash
# 1. 进入项目文件夹（注意 Git Bash 用 / 不用 \）
cd /c/Users/WONDER/.../herbal_identification

# 2. 初始化仓库
git init
git branch -M main

# 3. 添加所有文件
git add .

# 4. 提交
git commit -m "feat: 初始化中草药识别项目（训练/预测/配置）"

# 5. 在 GitHub 网页新建仓库 herbal-identification（不勾选 README）

# 6. 连接远程并推送
git remote add origin https://github.com/wonder-you-you/herbal-identification.git
git push -u origin main
```

---

## 注意事项
- `output/` 和数据集已被 `.gitignore` 忽略，**不会上传**（避免大文件）
- `best_model.pth` 模型权重较大，如需分享可用 Git LFS 或外部链接
- 上传后 README.md 会自动显示在仓库首页
