# ACT-Net

The official repository for *ACT-Net: Adaptive Compensation Transformer Network for Real-Time Detection in Remote Sensing Multimodal Imagery*.

This implementation is based on Ultralytics RT-DETR and includes VFE-Block, ACM, and PSA.

## Installation

Create a Python environment and install compatible versions of PyTorch and torchvision for your hardware. From the repository root, install the modified Ultralytics package and additional dependencies:

```bash
pip install -e .
pip install timm einops torchsummary mmengine
```

## Dataset

Prepare the dataset in Ultralytics detection format and create a configuration file such as `dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: class_0
  1: class_1
```

Replace the paths and class names with those of your dataset. `ACT-Net.yaml` defaults to five classes. The dataset configuration supplies the class count during training.

## Training

Run from the repository root:

```bash
python train.py --data /path/to/dataset.yaml --epochs 600 --imgsz 640 --batch 4
```

The training script builds the model from `ACT-Net.yaml`. Set data loader workers with `--workers` and adjust other training options in `train.py`. Results and weights are saved under `runs` by default.

## Validation

Validate a trained model using its weights and dataset configuration:

```python
from ultralytics import RTDETR

model = RTDETR("runs/detect/train/weights/best.pt")
metrics = model.val(data="/path/to/dataset.yaml", split="val", imgsz=640, batch=4)
print(metrics.box.map)  # mAP50-95
```

Update the weights path to match your training output. Set `split="test"` to evaluate the test set specified in the dataset configuration.

## License

This project contains code based on Ultralytics. See [LICENSE](LICENSE) for details.
