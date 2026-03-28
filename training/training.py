import torch
import torch.nn as nn
from tqdm import tqdm # библиотека для красивых прогресс-баров
import os
from sklearn.metrics import accuracy_score, f1_score # готовые функции для расчета метрик (accuracy, f1)

class Trainer:
    """Класс для обучения модели"""
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model # модель нейросети
        self.train_loader = train_loader #  загрузчик обучающих данных
        self.test_loader = test_loader # загрузчик тестовых данных
        self.config = config # объект с гиперпараметрами
        
        # Оптимизатор и функция потерь
        self.optimizer = torch.optim.Adam( # Adaptive Moment Estimation
            model.parameters(), # передаем все обучаемые веса модели
            lr=config.lr, # скорость обучения
            weight_decay=config.weight_decay # L2 регуляризация
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        ) # Каждые step_size=30 эпох умножает learning rate на gamma=0.1
        
        # Для логирования (метрики для построения графиков после обучения)
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self):
        """Одна эпоха обучения"""
        self.model.train() # переводим модель в режим обучения
        total_loss = 0
        
        for data, target in tqdm(self.train_loader, desc='Training'):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad() # Градиенты накапливаются, их нужно обнулять перед каждым шагом
            output = self.model(data) # Пропускаем данные через сеть
            loss = self.criterion(output, target) # Считаем ошибку
            
            # Backward pass
            loss.backward() # Вычисляем градиенты
            self.optimizer.step() # Обновляем веса
            
            total_loss += loss.item() # loss.item() - извлекаем число из тензора (убираем градиенты)
        
        return total_loss / len(self.train_loader) # средня потеря за эпоху
    
    def validate(self):
        """Валидация модели"""
        self.model.eval() # режим оценки: Отключает Dropout, BatchNorm использует накопленную статистику 
        all_preds = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad(): # Отключает вычисление градиенто, так как в валидации градиенты не нужны
            for data, target in tqdm(self.test_loader, desc='Validating'):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Получаем предсказания
                preds = output.argmax(dim=1) # выбираем класс с максимальной вероятностью, dim=1 - по каналу классов (размер [batch, 10])
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                total_loss += loss.item()
        
        # Считаем метрики
        accuracy = accuracy_score(all_targets, all_preds) # вычисляем процент правильных предсказаний
        f1 = f1_score(all_targets, all_preds, average='weighted') # взвешенный F1-score, учитывает дисбаланс классов
        
        return {
            'loss': total_loss / len(self.test_loader),
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self):
        """Полный цикл обучения"""

        for epoch in range(self.config.epochs):
            # Обучаем эпоху
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Валидация
            val_metrics = self.validate()
            self.test_losses.append(val_metrics['loss'])
            self.test_accuracies.append(val_metrics['accuracy'])
            
            # Обновляем learning rate
            self.scheduler.step()

            # Запоминаем последнюю точность
            final_accuracy = val_metrics['accuracy']
            
            # Печатаем прогресс
            print(f'Epoch {epoch+1}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}')
            print(f'  Val Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'  Val F1: {val_metrics["f1"]:.4f}')
            
            # Сохраняем модель после всех эпох
        self.save_checkpoint(accuracy=final_accuracy)
        print(f'Model saved after {self.config.epochs} epochs')
    
    def save_checkpoint(self, accuracy=None):
        """Сохраняет чекпоинт модели"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(), # веса модели (словарь параметров)
            'optimizer_state_dict': self.optimizer.state_dict(), # состояние оптимизатора (momentum, lr)
            'accuracy': accuracy,
            'config': self.config,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
        path = os.path.join(self.config.checkpoint_dir, 'model.pth')
        torch.save(checkpoint, path)
    
    def load_model(self):
        """Загружает сохраненную модель"""
        path = os.path.join(self.config.checkpoint_dir, 'model.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.config.device) # map_location=self.config.device позволяет загрузить модель, сохраненную на GPU, на CPU
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print("No checkpoint found")