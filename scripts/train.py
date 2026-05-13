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


def validate_dataset_dir(data_dir: Path):
    """验证数据集目录结构是否完整"""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在: {data_dir}\n"
            f"请确保目录结构为:\n"
            f"  {data_dir}/train/good/   (正常样本,训练用)\n"
            f"  {data_dir}/test/good/    (正常样本,测试用)\n"
            f"  {data_dir}/test/bad/     (缺陷样本,测试用)"
        )

    required_dirs = [
        data_dir / "train" / "good",
        data_dir / "test" / "good",
        data_dir / "test" / "bad",
    ]
    missing = [d for d in required_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"数据集目录结构不完整,缺少以下目录:\n"
            + "\n".join(f"  {d}" for d in missing)
        )


def train(model_name: str = None, data_dir: str = None,
          image_size: int = 256, max_epochs: int = 1):
    """
    训练缺陷检测模型

    训练流程:
    1. 初始化模型
    2. 加载数据集
    3. 训练模型
    4. 评估模型性能

    Args:
        model_name: 模型名称 ('padim' 或 'patchcore')
        data_dir: 数据集根目录路径
        image_size: 输入图片尺寸
        max_epochs: 训练轮数

    Returns:
        训练完成的模型实例
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    if data_dir is None:
        data_dir = PROJECT_ROOT / "datasets"
    else:
        data_dir = Path(data_dir)

    validate_dataset_dir(data_dir)

    model_info = get_model_info(model_name)

    print("=" * 50)
    print(f"{model_name.upper()} 缺陷检测模型训练")
    print("=" * 50)
    print(f"模型: {model_info['description']}")
    print(f"骨干网络: {model_info['backbone']}")
    print(f"特征层: {model_info['layers']}")
    print(f"图片尺寸: {image_size}x{image_size}")
    print(f"训练轮数: {max_epochs}")

    model = get_model(model_name)

    datamodule = Folder(
        name="defect_detection",
        root=str(data_dir),
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        image_size=(image_size, image_size),
    )

    engine = Engine(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=str(PROJECT_ROOT / "outputs"),
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
    
    print(f"\n模型已保存至: {PROJECT_ROOT / 'outputs'}")
    
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集根目录路径,默认: ./datasets",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="输入图片尺寸 (默认: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="训练轮数 (默认: 1)",
    )

    args = parser.parse_args()
    train(args.model, args.data_dir, args.image_size, args.epochs)
