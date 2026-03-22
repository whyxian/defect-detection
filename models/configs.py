"""
模型配置模块

集中管理所有缺陷检测模型的配置参数,便于切换和扩展。
支持的模型:
- padim: PaDiM 模型,基于多元高斯分布
- patchcore: PatchCore 模型,基于记忆库最近邻检索
"""

from anomalib.models import Padim, Patchcore


MODEL_CONFIGS = {
    "padim": {
        "class": Padim,
        "backbone": "resnet18",
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
        "description": "PaDiM: 基于多元高斯分布的无监督异常检测",
    },
    "patchcore": {
        "class": Patchcore,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "pre_trained": True,
        "num_neighbors": 9,
        "description": "PatchCore: 基于记忆库最近邻检索的无监督异常检测",
    },
}


DEFAULT_MODEL = "padim"


AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())
