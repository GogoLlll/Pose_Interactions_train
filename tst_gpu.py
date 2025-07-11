import torch

print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Кол-во доступных GPU:", torch.cuda.device_count())
    print("Имя текущего GPU:", torch.cuda.get_device_name(0))
    print("Версия CUDA:", torch.version.cuda)
    print("Выделенная память:", torch.cuda.memory_reserved(0) / 1024**3, "GB")
    print("Используемая память:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
else:
    print("GPU не обнаружен. Используется CPU.")