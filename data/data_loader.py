import torch # основной фреймворк, нужен для работы с тензорами
from torch.utils.data import DataLoader # класс для создания батчей, перемешивания данных, параллельной загрузки
from torchvision import datasets, transforms # datasets содержит готовые датасеты (MNIST); transforms - инструменты для преобразования изображений (нормализация, аугментации)

def get_transforms():
    """Возвращает transform для train и test"""
    # Базовые преобразования для всех данных
    base_transform = transforms.Compose([
        transforms.ToTensor(),  # Превращаем в тензор [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Нормализация для MNIST:  output = (input - mean) / std
    ])
    
    # Аугментации для обучения
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # вращение на угол от -10° до +10°; сдвиг по горизонтали и вертикали до 10% от размера изображения 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, base_transform

def get_data_loaders(config):
    """Создает загрузчики данных для train/test"""
    # Получаем трансформы
    train_transform, test_transform = get_transforms()
    
    # Загружаем датасеты
    if config.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            root='./data', train=True, 
            download=True, transform=train_transform # download=True - автоматически скачать, если нет локально
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False,
            download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Создаем загрузчики
    train_loader = DataLoader( # класс управляет подачей данных в модель
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, #перемешивает данные перед каждой эпохой, чтобы модель не запоминала порядок примеров
        pin_memory=True if config.device.type == 'cuda' else False  # Ускоряет transfer на GPU
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False  # Ускоряет transfer на GPU
    )
    
    return train_loader, test_loader

def get_sample_images(test_loader, num_images=10):
    """Отбирает несколько изображений для визуализации"""
    images, labels = next(iter(test_loader))
    # Берем первые num_images штук
    return images[:num_images], labels[:num_images]