"""
Task 2: Model Training Script for PPE Compliance.
Fine-tunes YOLOv8n at 416px resolution with CCTV-specific augmentations.
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def create_synthetic_ppe_data_if_empty(dataset_root: str = "data/ppe"):
    """Creates synthetic placeholder images and labels if dataset folder is empty."""
    train_img_dir = os.path.join(dataset_root, "train", "images")
    train_lbl_dir = os.path.join(dataset_root, "train", "labels")
    val_img_dir = os.path.join(dataset_root, "val", "images")
    val_lbl_dir = os.path.join(dataset_root, "val", "labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    existing_train = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))])
    if existing_train >= 10:
        return

    print(f"[Notice] Generating synthetic sample PPE dataset in '{dataset_root}' for verification...")
    # Generate 20 train and 10 val images
    for split, count in [("train", 20), ("val", 10)]:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")

        for i in range(count):
            img = np.full((416, 416, 3), 120, dtype=np.uint8)
            # Add synthetic worker head with helmet (class 0: helmet, yellow)
            cv2.circle(img, (150, 150), 35, (0, 220, 255), -1)
            # Add synthetic worker head without helmet (class 1: no_helmet, hair/skin tone)
            cv2.circle(img, (280, 200), 30, (80, 110, 160), -1)

            img_name = f"sample_{i:03d}.jpg"
            lbl_name = f"sample_{i:03d}.txt"
            cv2.imwrite(os.path.join(img_dir, img_name), img)

            # YOLO format: class_id center_x center_y width height (normalized)
            with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                f.write(f"0 {150/416:.4f} {150/416:.4f} {70/416:.4f} {70/416:.4f}\n")
                f.write(f"1 {280/416:.4f} {200/416:.4f} {60/416:.4f} {60/416:.4f}\n")

    print("[Notice] Synthetic PPE dataset created.")


def train_model(
    data_yaml: str = "data/ppe/data.yaml",
    epochs: int = 15,
    imgsz: int = 416,
    batch_size: int = 16,
    device: str = "0",
    output_dir: str = "task2/outputs",
):
    print("=" * 70)
    print("Task 2: Fine-Tuning YOLOv8n for Factory PPE Compliance")
    print("=" * 70)

    create_synthetic_ppe_data_if_empty()

    # Load baseline YOLOv8n COCO weights
    model = YOLO("yolov8n.pt")

    # Hyperparameters tailored for CCTV camera perspectives:
    # - mosaic=1.0: mixes 4 images, simulates multi-worker scenes
    # - mixup=0.1: blend images to handle occlusions
    # - hsv_h=0.015, hsv_s=0.5, hsv_v=0.3: light variations in warehouse/factory
    # - degrees=5.0: slight tilt of cameras
    # - translate=0.1: shift workers across field of view
    # - scale=0.3: workers at different distances from camera
    # - perspective=0.0005: CCTV oblique view distortion
    # - flipud=0.0: workers are never upside-down
    # - fliplr=0.5: left-right horizontal symmetry
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=output_dir,
        name="ppe_yolov8n",
        exist_ok=True,
        # CCTV augmentations
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        plots=True,
    )

    best_pt_path = os.path.join(output_dir, "ppe_yolov8n", "weights", "best.pt")
    target_best = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_pt_path):
        import shutil
        shutil.copy(best_pt_path, target_best)
        print(f"[Success] Best model weights copied to: {target_best}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on PPE dataset")
    parser.add_argument("--data", type=str, default="data/ppe/data.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="task2/outputs")

    args = parser.parse_args()
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_dir=args.output_dir,
    )
