import json

json_path = 'dataset_2/labels/train/instances_train.json'

try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=== Общая структура JSON ===")
    print(f"Всего секций: {len(data.keys())}")
    for key in data.keys():
        print(f"- Секция: {key}, Количество записей: {len(data[key])}")

    print("\n=== Секция 'images' ===")
    print(f"Общее количество изображений: {len(data['images'])}")
    print("Пример первой записи:")
    print(json.dumps(data['images'][0], indent=2))
    print("Поля в первой записи:", list(data['images'][0].keys()))
    print("Уникальные поля во всех записях:", {k for img in data['images'] for k in img.keys()})

    print("\n=== Секция 'annotations' ===")
    print(f"Общее количество аннотаций: {len(data['annotations'])}")
    print("Пример первой аннотации:")
    print(json.dumps(data['annotations'][0], indent=2))
    print("Поля в первой аннотации:", list(data['annotations'][0].keys()))
    print("Уникальные поля во всех аннотациях:", {k for ann in data['annotations'] for k in ann.keys()})

    print("\n=== Секция 'categories' ===")
    print(f"Общее количество категорий: {len(data['categories'])}")
    print("Пример категории:")
    print(json.dumps(data['categories'][0], indent=2))
    print("Поля в категории:", list(data['categories'][0].keys()))

except FileNotFoundError:
    print(f"Файл {json_path} не найден. Проверьте путь.")
except json.JSONDecodeError as e:
    print(f"Ошибка декодирования JSON: {e}. Проверьте валидность файла.")
except Exception as e:
    print(f"Произошла ошибка: {e}")