# Pose_Interactions_train

## Структура проекта

```
Pose_Interactions_train/
├── train.py # Основной файл запуска обучения
├── requirements.txt # Все зависимости
├── yolo11x-pose.pt # Базовая модель
├── tst_gpu.py
├── dataset_1/
│ ├── images/
│ │ ├── train/
│ │ └── val/
│ └── annotations/
│ │ ├── train/
│ │ └── val/
│ │ crowdpose.yaml 
```

Для установки зависимостей напишите
```
pip install -r requirements.txt
```

После скачайте архив с датасетом, а так же модель для дообучения по
[ссылке](https://disk.360.yandex.ru/d/EWEl_f9-zxpvYQ) и после распакуйте все как в струкутре выше.
После запустите `tst_gpu.py`, чтобы проверить видит ли окружение GPU, если нет, то скачайте PyTorch с поддержкой cuda на офф. сайте.

В файле `crowdpose.yaml `
```
path: C:\Users\garni\PycharmProjects\Pose_study_UseTech\dataset_1

train: images/train
val: images/val

kpt_shape: [17, 3]

flip_idx: [0, 2, 1, 5, 4, 3, 7, 6, 8, 12, 11, 10, 9, 14, 13, 16, 15]
names:
  0: person
```
Поставьте правильный абсолютный путь в первой строке `path`.

### Запуск обучения

Нужно запустить файл `train.py`
```
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
```


