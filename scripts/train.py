"""
PaDiM 缺陷检测模型训练脚本
使用 anomalib 2.1.0 API
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
