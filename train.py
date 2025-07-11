from ultralytics import YOLO

def train_yolo_pose():
    model = YOLO("yolo11x-pose.pt")

    model.train(
        data="dataset_1/crowdpose.yaml",
        epochs=100,
        imgsz=960,
        batch=32,
        lr0=0.002,
        momentum=0.9,
        weight_decay=0.0005,
        freeze=12,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        save_period=5,
        workers=12,
        project="runs/pose",
        name="crowdpose_finetune_v1",
        val=True,
        visualize=True,
    )

if __name__ == "__main__":
    train_yolo_pose()