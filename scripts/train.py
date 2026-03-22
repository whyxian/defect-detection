"""
PaDiM 缺陷检测模型训练脚本
使用 anomalib 2.1.0 API

PaDiM 是一种基于嵌入向量的无监督异常检测方法:
1. 使用预训练的 CNN 提取多层级特征
2. 用多元高斯分布对正常样本特征进行建模
3. 推理时计算测试样本与高斯分布的马氏距离作为异常分数
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim


PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"
OUTPUT_PATH = PROJECT_ROOT / "outputs"


def train():
    """
    训练 PaDiM 缺陷检测模型
    
    训练流程:
    1. 初始化 PaDiM 模型
    2. 加载数据集
    3. 训练模型（估计特征分布参数）
    4. 评估模型性能
    
    Returns:
        Padim: 训练完成的模型实例
    """
    print("=" * 50)
    print("PaDiM 缺陷检测模型训练")
    print("=" * 50)
    
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )
    
    datamodule = Folder(
        name="defect_detection",
        root=DATASET_PATH,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
    )
    
    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        default_root_dir=OUTPUT_PATH,
    )
    
    print(f"\n开始训练...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )
    
    print(f"\n训练完成!")
    print(f"\n评估模型性能...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )
    
    print(f"\n测试结果:")
    for result in test_results:
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
    
    print(f"\n模型已保存至: {OUTPUT_PATH}")
    
    return model


if __name__ == "__main__":
    train()
