"""
缺陷检测模型训练脚本
使用 anomalib 2.1.0 API

支持通过命令行参数切换模型:
    python train.py                    # 使用默认模型 (padim)
    python train.py --model patchcore  # 使用 PatchCore 模型
    python train.py --model padim      # 使用 PaDiM 模型
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from anomalib.data import Folder
from anomalib.engine import Engine

from models import get_model, get_model_info, AVAILABLE_MODELS, DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"
OUTPUT_PATH = PROJECT_ROOT / "outputs"


def train(model_name: str = None):
    """
    训练缺陷检测模型
    
    训练流程:
    1. 初始化模型
    2. 加载数据集
    3. 训练模型
    4. 评估模型性能
    
    Args:
        model_name: 模型名称 ('padim' 或 'patchcore')
    
    Returns:
        训练完成的模型实例
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    model_info = get_model_info(model_name)
    
    print("=" * 50)
    print(f"{model_name.upper()} 缺陷检测模型训练")
    print("=" * 50)
    print(f"模型: {model_info['description']}")
    print(f"骨干网络: {model_info['backbone']}")
    print(f"特征层: {model_info['layers']}")
    
    model = get_model(model_name)
    
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
    parser = argparse.ArgumentParser(description="缺陷检测模型训练")
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f"模型名称,可选: {AVAILABLE_MODELS}, 默认: {DEFAULT_MODEL}",
    )
    
    args = parser.parse_args()
    train(args.model)
