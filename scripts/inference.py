"""
PatchCore 缺陷检测推理脚本
使用 anomalib 2.1.0 API

支持两种推理模式:
1. 单张图片检测: python inference.py --image /path/to/image.png
2. 批量检测: python inference.py --dir /path/to/images/

输出结果包含:
- 异常分数 (0-1, 越高越异常)
- 判定结果 (正常/缺陷)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "outputs"


def find_checkpoint():
    """
    查找训练好的模型检查点文件
    
    模型保存路径结构:
    outputs/Patchcore/defect_detection/v0/weights/lightning/*.ckpt
    
    Returns:
        Path | None: 检查点文件路径, 未找到则返回 None
    """
    ckpt_dir = OUTPUT_PATH / "Patchcore" / "defect_detection" / "v0" / "weights" / "lightning"
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            return ckpt_files[0]
    return None


def predict_single(image_path: str):
    """
    对单张图片进行缺陷检测
    
    Args:
        image_path: 图片文件路径
    
    Returns:
        list: 预测结果列表
    
    Raises:
        FileNotFoundError: 图片不存在或模型未训练
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    print("=" * 50)
    print("PatchCore 缺陷检测推理")
    print("=" * 50)
    
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )
    
    engine = Engine()
    
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("未找到训练好的模型，请先运行 train.py 进行训练")
    
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


def predict_batch(image_dir: str):
    """
    批量检测目录下的所有图片
    
    Args:
        image_dir: 图片目录路径
    
    Returns:
        list: 预测结果列表
    
    Raises:
        FileNotFoundError: 目录不存在或模型未训练
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
    print("=" * 50)
    print("PatchCore 批量缺陷检测")
    print("=" * 50)
    
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )
    
    engine = Engine()
    
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("未找到训练好的模型，请先运行 train.py 进行训练")
    
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
    parser = argparse.ArgumentParser(description="PatchCore 缺陷检测推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片目录路径（批量检测）")
    
    args = parser.parse_args()
    
    if args.image:
        predict_single(args.image)
    elif args.dir:
        predict_batch(args.dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python inference.py --image /path/to/image.png")
        print("  python inference.py --dir /path/to/images/")
