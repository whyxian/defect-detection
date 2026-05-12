# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

工业缺陷检测项目,使用 [anomalib](https://github.com/openvinotoolkit/anomalib) 2.1.0 实现无监督异常检测。支持 PaDiM 和 PatchCore 两种模型。

## Commands

```bash
# 训练模型 (默认 padim)
python scripts/train.py --model padim
python scripts/train.py --model patchcore

# 单张图片推理
python scripts/inference.py --image /path/to/image.png --model padim

# 批量推理
python scripts/inference.py --dir /path/to/images/ --model patchcore
```

## Project Structure

```
defect_detection/
├── models/                    # 模型工厂模块
│   ├── configs.py             # 模型配置 (padim/patchcore), 骨干网络, 特征层
│   ├── factory.py             # get_model(), get_checkpoint_path(), get_model_info()
│   └── __init__.py            # 公开 API
├── scripts/
│   ├── train.py               # 训练脚本 (Folder datamodule, Engine.fit + Engine.test)
│   └── inference.py           # 推理脚本 (单张 + 批量, PredictDataset, Engine.predict)
├── datasets/                  # 数据集 (gitignored)
│   ├── train/good/            # 正常样本 (训练)
│   ├── test/good/             # 正常样本 (测试)
│   └── test/bad/              # 缺陷样本 (测试)
├── outputs/                   # 模型检查点 (gitignored)
│   └── {ModelName}/defect_detection/v{version}/weights/lightning/model.ckpt
└── results/                   # 推理结果 (gitignored)
```

## Architecture

- **Model Factory 模式**: 通过 `models/configs.py` 集中管理模型参数,`models/factory.py` 根据名称创建模型实例。新增模型只需在 `MODEL_CONFIGS` 中添加配置条目。
- **训练流程**: `Folder` datamodule 加载数据集 → `Engine(max_epochs=1)` 训练 → 自动评估。
- **推理流程**: `PredictDataset` 加载图片 → `Engine.predict()` 推理 → 返回 `pred_label`(0/1) 和 `pred_score`(0-1 异常分数)。
- **检查点路径**: `outputs/{ClassName}/defect_detection/v{version}/weights/lightning/*.ckpt`。

## Environment

- **Conda 环境**: `defect_detection` (`/home/xian/miniconda3/envs/defect_detection`)
- **Python 版本**: 3.10.20
- **CUDA**: 12.9 (cuda-bindings 12.9.4)
- **激活方式**: `conda activate defect_detection`
- **环境文件**: `environment.yml` (conda) / `requirements.txt` (pip)

## Key Dependencies

| 包 | 版本 | 用途 |
|---|------|------|
| anomalib | 2.1.0 | 无监督异常检测框架 |
| torch | 2.10.0 | 深度学习框架 |
| torchvision | 0.25.0 | 图像处理 |
| lightning / pytorch-lightning | 2.6.1 | 训练引擎 |
| timm | 1.0.25 | 骨干网络 (ResNet/WideResNet) |
| einops | 0.8.2 | 张量操作 |
| opencv-python | 4.13.0 | 图像 IO 与预处理 |
| scikit-learn | 1.7.2 | 评估指标 |
| scikit-image | 0.25.2 | 图像后处理 |
| kornia | 0.8.2 | 可微图像处理 |
| matplotlib | 3.10.8 | 可视化 |
| tensorboard | 2.20.0 | 训练日志可视化 |
| pandas | 2.3.3 | 结果数据分析 |
