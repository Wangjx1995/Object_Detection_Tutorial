import argparse, yaml
from ultralytics import YOLO
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./configs/dataset.yaml", help="dataset yaml")
    ap.add_argument("--cfg",  default="./configs/train.yaml",   help="training config yaml")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

 
    data_yaml = args.data

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    device = "0" if torch.cuda.is_available() else "cpu"
    results = model.train(
        data=data_yaml,
        imgsz=cfg.get("imgsz", 512),
        epochs=cfg.get("epochs", 15),
        batch=cfg.get("batch", 16),
        patience=cfg.get("patience", 5),
        workers=cfg.get("workers", 2),
        device=cfg.get("device", 0),
        project=cfg.get("project", "runs"),
        name=cfg.get("name", "train"),
        cache=cfg.get("cache", "ram"),
        plots=cfg.get("plots", True),
        seed=cfg.get("seed", 42),
    )
    print("Saved to:", results.save_dir)

if __name__ == "__main__":
    main()