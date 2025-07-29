from ultralytics import YOLO

def train_model():
    model = YOLO('yolo11l-pose.pt')

    training_params = {
        'data': 'dataset_3/data.yaml',
        'epochs': 50,
        'imgsz': 960,
        'batch': 8,
        'dropout': 0.1,
        'weight_decay': 0.0005,
        'lr0': 0.001,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'augment': True,
        'device': 0,
        'project': 'runs/train',
        'name': 'yolo11l_pose_exp',
        'exist_ok': True,
        'pretrained': True,
        'val': True
    }

    model.train(**training_params)

    model.val(data=training_params['data'], imgsz=training_params['imgsz'], batch=training_params['batch'])

if __name__ == '__main__':
    train_model()
