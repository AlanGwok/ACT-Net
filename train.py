from pathlib import Path
import argparse
from ultralytics import RTDETR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ACT-Net')
    parser.add_argument('--data', required=True, help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    model = RTDETR(str(Path(__file__).resolve().parent / 'ACT-Net.yaml'))
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, optimizer='AdamW', lr0=0.0001, lrf=0.1,
                momentum=0.9, weight_decay=1e-5, warmup_epochs=3,
                warmup_momentum=0.8, warmup_bias_lr=0.1, amp=False, patience=70)
