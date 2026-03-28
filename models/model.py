import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """Сверточная нейросеть для классификации MNIST"""
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()

        self.conv5_block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
)
        
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Второй сверточный блок
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Третий сверточный блок (для Grad-CAM нам нужен последний conv слой)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)  # 7x7 -> 3x3
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 128*9 = 1152 -> 256
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.feature_maps = None
        
        # Считаем параметры (для проверки лимита)
        self.print_params_count()
    
    def forward(self, x):
        # Первый блок
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Второй блок
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Третий блок (сохраняем активации для Grad-CAM)
        x = self.conv5_block(x)
        self.feature_maps = x
 
        
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Полносвязная часть
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self):
        # Возвращает карты признаков последнего conv слоя
        return self.feature_maps
    
    def print_params_count(self):
        """Выводит количество параметров модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        if total_params < 5_000_000:
            print(f'Параметров в пределах лимита (5 млн)')
        else:
            print(f'Превышен лимит параметров!')