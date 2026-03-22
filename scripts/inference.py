"""
缺陷检测推理脚本
使用 anomalib 2.1.0 API

支持两种推理模式:
1. 单张图片检测: python inference.py --image /path/to/image.png
2. 批量检测: python inference.py --dir /path/to/images/

支持通过命令行参数切换模型:
    python inference.py --image test.png --model patchcore
    python inference.py --dir images/ --model padim

输出结果包含:
- 异常分数 (0-1, 越高越异常)
- 判定结果 (正常/缺陷)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from anomalib.data import PredictDataset
from anomalib.engine import Engine

from models import (
    get_model,
    get_checkpoint_path,
    get_model_info,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "outputs"


def predict_single(image_path: str, model_name: str = None):
    """
    对单张图片进行缺陷检测
    
    Args:
        image_path: 图片文件路径
        model_name: 模型名称 ('padim' 或 'patchcore')
    
    Returns:
        list: 预测结果列表
    
    Raises:
        FileNotFoundError: 图片不存在或模型未训练
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    model_info = get_model_info(model_name)
    
    print("=" * 50)
    print(f"{model_name.upper()} 缺陷检测推理")
    print("=" * 50)
    print(f"模型: {model_info['description']}")
    
    model = get_model(model_name)
    engine = Engine()
    
    ckpt_path = get_checkpoint_path(model_name, OUTPUT_PATH)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"未找到 {model_name} 模型,请先运行: python train.py --model {model_name}"
        )
    
    print(f"\n加载模型: {ckpt_path}")
    print(f"\n检测图片: {image_path}")
    
    dataset = PredictDataset(
        path=image_path.parent,
        image_size=(256, 256),
    )
    
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )
    
    if predictions is not None:
        for prediction in predictions:
            pred_label = prediction.pred_label
            pred_score = prediction.pred_score.item() if hasattr(prediction.pred_score, 'item') else prediction.pred_score
            
            print(f"\n检测结果:")
            print(f"  - 图片: {image_path.name}")
            print(f"  - 异常分数: {pred_score:.4f}")
            
            if pred_label:
                print(f"  - 判定: 缺陷图片 ❌")
            else:
                print(f"  - 判定: 正常图片 ✅")
    
    return predictions


def predict_batch(image_dir: str, model_name: str = None):
    """
    批量检测目录下的所有图片
    
    Args:
        image_dir: 图片目录路径
        model_name: 模型名称 ('padim' 或 'patchcore')
    
    Returns:
        list: 预测结果列表
    
    Raises:
        FileNotFoundError: 目录不存在或模型未训练
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
    model_info = get_model_info(model_name)
    
    print("=" * 50)
    print(f"{model_name.upper()} 批量缺陷检测")
    print("=" * 50)
    print(f"模型: {model_info['description']}")
    
    model = get_model(model_name)
    engine = Engine()
    
    ckpt_path = get_checkpoint_path(model_name, OUTPUT_PATH)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"未找到 {model_name} 模型,请先运行: python train.py --model {model_name}"
        )
    
    print(f"\n加载模型: {ckpt_path}")
    
    dataset = PredictDataset(
        path=image_dir,
        image_size=(256, 256),
    )
    
    print(f"\n开始批量检测...")
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )
    
    print(f"\n检测结果汇总:")
    print("-" * 50)
    
    normal_count = 0
    defect_count = 0
    
    if predictions is not None:
        for prediction in predictions:
            image_path = prediction.image_path
            pred_label = prediction.pred_label
            pred_score = prediction.pred_score.item() if hasattr(prediction.pred_score, 'item') else prediction.pred_score
            
            status = "❌ 缺陷" if pred_label else "✅ 正常"
            image_name = Path(image_path).name if isinstance(image_path, str) else str(image_path)
            print(f"  {image_name}: 异常分数={pred_score:.4f} {status}")
            
            if pred_label:
                defect_count += 1
            else:
                normal_count += 1
    
    print("-" * 50)
    print(f"统计: 正常 {normal_count} 张, 缺陷 {defect_count} 张")
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缺陷检测推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片目录路径（批量检测）")
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f"模型名称,可选: {AVAILABLE_MODELS}, 默认: {DEFAULT_MODEL}",
    )
    
    args = parser.parse_args()
    
    if args.image:
        predict_single(args.image, args.model)
    elif args.dir:
        predict_batch(args.dir, args.model)
    else:
        parser.print_help()
        print("\n示例:")
        print(f"  python inference.py --image /path/to/image.png --model {DEFAULT_MODEL}")
        print(f"  python inference.py --dir /path/to/images/ --model patchcore")
