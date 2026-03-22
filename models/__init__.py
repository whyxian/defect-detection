"""
模型模块

提供统一的模型创建和管理接口。

使用示例:
    from models import get_model, get_checkpoint_path, get_model_info
    
    # 创建模型
    model = get_model("patchcore")
    
    # 获取检查点路径
    ckpt_path = get_checkpoint_path("patchcore")
    
    # 查看模型信息
    info = get_model_info()
"""

from .factory import get_model, get_checkpoint_path, get_model_info
from .configs import MODEL_CONFIGS, DEFAULT_MODEL, AVAILABLE_MODELS

__all__ = [
    "get_model",
    "get_checkpoint_path",
    "get_model_info",
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
]
