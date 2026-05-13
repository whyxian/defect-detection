"""
模型工厂模块

提供统一的模型创建和检查点路径获取接口。
"""

from pathlib import Path

from .configs import MODEL_CONFIGS, DEFAULT_MODEL, AVAILABLE_MODELS


# 配置中不作为模型构造参数的元数据键
_META_KEYS = {"class", "description"}


def get_model(model_name: str = None):
    """
    根据模型名称创建模型实例

    Args:
        model_name: 模型名称,可选值: 'padim', 'patchcore'
                   如果为 None,则使用默认模型

    Returns:
        模型实例 (Padim 或 Patchcore)

    Raises:
        ValueError: 模型名称不存在时抛出

    Example:
        >>> model = get_model("patchcore")
        >>> model = get_model()  # 使用默认模型
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    model_name = model_name.lower()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"未知模型: '{model_name}'. "
            f"可用模型: {AVAILABLE_MODELS}"
        )

    config = MODEL_CONFIGS[model_name]
    model_class = config["class"]
    # 将非元数据的配置项作为模型构造参数传入
    model_params = {
        key: value
        for key, value in config.items()
        if key not in _META_KEYS
    }

    return model_class(**model_params)


def _find_latest_version_dir(model_dir: Path) -> Path | None:
    """
    查找模型最新的版本目录 (v0, v1, v2...)

    Args:
        model_dir: 模型输出目录

    Returns:
        Path | None: 最新版本目录下的 weights/lightning 子目录,未找到则返回 None
    """
    if not model_dir.exists():
        return None

    version_dirs = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("v")
    ]
    if not version_dirs:
        return None

    # 按版本号降序排列,取最新版本
    latest = max(version_dirs, key=lambda d: int(d.name[1:]))
    ckpt_dir = latest / "weights" / "lightning"
    return ckpt_dir if ckpt_dir.exists() else None


def get_checkpoint_path(model_name: str = None, output_dir: Path = None):
    """
    获取模型检查点文件路径
    
    Args:
        model_name: 模型名称,如果为 None 则使用默认模型
        output_dir: 输出目录路径,如果为 None 则使用默认路径
    
    Returns:
        Path | None: 检查点文件路径,未找到则返回 None
    
    Example:
        >>> ckpt_path = get_checkpoint_path("patchcore")
        >>> ckpt_path = get_checkpoint_path()  # 使用默认模型
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    model_name = model_name.lower()

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"

    model_class_name = MODEL_CONFIGS[model_name]["class"].__name__

    model_dir = output_dir / model_class_name / "defect_detection"
    ckpt_dir = _find_latest_version_dir(model_dir)

    if ckpt_dir is not None:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            return ckpt_files[0]

    return None


def get_model_info(model_name: str = None):
    """
    获取模型信息
    
    Args:
        model_name: 模型名称,如果为 None 则返回所有模型信息
    
    Returns:
        dict: 模型配置信息字典
    
    Example:
        >>> info = get_model_info("patchcore")
        >>> print(info["description"])
    """
    if model_name is None:
        return {
            name: {
                "backbone": config["backbone"],
                "layers": config["layers"],
                "description": config.get("description", ""),
            }
            for name, config in MODEL_CONFIGS.items()
        }
    
    model_name = model_name.lower()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"未知模型: '{model_name}'. "
            f"可用模型: {AVAILABLE_MODELS}"
        )
    
    config = MODEL_CONFIGS[model_name]
    return {
        "backbone": config["backbone"],
        "layers": config["layers"],
        "description": config.get("description", ""),
    }
