import json

# 实际检测到的类别列表（共72个，均为英文）
actual_classes = [
    "Anxixiang", "Baibiandou", "Baifan", "Bailian", "Baimaogen",
    "Baiqian", "Baishao", "Baizhi", "Baiziren", "Beishashen",
    "Bibo", "Bichengqie", "Biejia", "Binglang", "Cangzhu",
    "Caodoukou", "Chenxiang", "Chuanlianzi", "Chuanmuxiang",
    "Chuanniuxi", "Dafupi", "Dandouchi", "Daoya", "Dilong",
    "Dongchongxiacao", "Fangfeng", "Fanxieye", "Fengfang",
    "Gancao", "Ganjiang", "Gansong", "Gaoben", "Ghishizhi",
    "Gouqizi", "Guizhi", "Gujingcao", "Guya", "HaiIong",
    "Haipiaoxiao", "Hehuanpi", "Huangbo", "Huangqi", "Huangqin",
    "Hubeibeimu", "Jiangcan", "Jiezi", "Jiguanhua", "Jindenglong",
    "Jineijin", "Jingjiesui", "Jinguolan", "Jinqianbaihuashe",
    "Jiuxiangchong", "Juhe", "Kudiding", "Laifuzi", "Lianfang",
    "Lianxu", "Lianzi", "Lianzixin", "Lingzhi", "Lizhihe",
    "Longyanrou", "Lugen", "Lulutong", "Maidong", "Mudingxiang",
    "Qianghuo", "Qiannianjian", "Qinpi", "Quanxie", "Rendongteng"
]


def generate_class_codes(classes):
    """为英文类别生成代码（首字母大写+数字编号）"""
    codes = []
    for idx, cls_name in enumerate(classes, 1):
        # 英文类别处理：首字母大写+3位数字
        # 例如：Anxixiang -> Axi001
        code = f"{cls_name[0].upper()}{cls_name[1:3].lower()}{idx:03d}"
        codes.append(code)
    return codes


# 生成配置数据
config_data = {
    "classes": actual_classes,  # 类别名称列表
    "class_codes": generate_class_codes(actual_classes),  # 自动生成的类别代码
    "img_size": 224  # 模型输入图像尺寸
}

# 保存配置文件
with open("config.json", "w", encoding='utf-8') as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)
    # indent=2: 美化输出，每行缩进2个空格
    # ensure_ascii=False: 正确保存非ASCII字符


# 验证配置文件
def validate_config(config_path):
    """验证配置文件的正确性"""
    with open(config_path) as f:
        config = json.load(f)

    # 验证类别和代码数量匹配
    assert len(config['classes']) == len(config['class_codes']), "类别与代码数量不匹配"
    # 验证类别总数是否符合预期
    assert len(config['classes']) == 72, "检测到异常类别数量"

    print("✅ 配置文件验证通过")
    print(f"总类别数: {len(config['classes'])}")
    print("前5个类别-代码对照：")
    for cls, code in zip(config['classes'][:5], config['class_codes'][:5]):
        # 对齐输出，便于查看
        print(f"{cls:15} -> {code}")


if __name__ == "__main__":
    # 验证生成的配置文件
    validate_config("config.json")
