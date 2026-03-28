import random
import numpy as np
import torch
import os

# Фиксируем seed для воспроизводимости
def set_seed(seed=42):
    """Устанавливает seed для всех библиотек"""
    random.seed(seed)  # Фиксируем random
    np.random.seed(seed)  # Фиксируем numpy
    torch.manual_seed(seed)  # Фиксируем генератор PyTorch на CPU torch
    torch.cuda.manual_seed_all(seed)  # Фиксируем генератор PyTorch на всех GPU
    torch.backends.cudnn.deterministic = True  # заставляет cuDNN (библиотеку для CUDA) использовать детерминированные алгоритмы.
    torch.backends.cudnn.benchmark = False  # Отключаем автооптимизацию
    os.environ['PYTHONHASHSEED'] = str(seed)  # Фиксируем хеширование Python

class Config:
    """Класс с конфигурацией эксперимента"""
    def __init__(self):
        # Параметры эксперимента
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # определяет, на каком устройстве будет выполняться вычисления
        
        # Параметры данных
        self.dataset = 'mnist'  # mnist 
        self.img_size = 28
        self.num_classes = 10
        self.batch_size = 256 # сколько изображений одновременно подается в модель
        self.num_workers = 2  # Для загрузки данных: количество процессов, которые параллельно загружают данные
        
        # Параметры обучения
        self.epochs = 100
        self.lr = 1e-3 #Скорость обучения - размер шага при обновлении весов модели.
        self.weight_decay = 1e-5  # L2 регуляризация - штраф за большие веса модели. Помогает бороться с переобучением
        
        # Пути для сохранения
        self.checkpoint_dir = 'checkpoints' # папка, куда сохраняются веса обученной модели
        self.results_dir = 'results' # папка, куда сохраняются все результаты: графики, метрики, визуализации
        
        # Параметры для визуализации
        self.num_test_images = 10  # Сколько картинок анализируем