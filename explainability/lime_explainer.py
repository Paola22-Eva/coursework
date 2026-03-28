import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift

class LIMEExplainer:
    """Обертка для LIME объяснений"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_proba(self, images):
        """Функция предсказания для LIME (нужна для совместимости)"""
        self.model.eval()
        batch_images = []
        
        for img in images:
            # Создаем копию, чтобы не изменять исходный массив
            img_processed = img.copy()
            
            # Конвертируем в uint8 если нужно
            if img_processed.max() <= 1:
                img_processed = (img_processed * 255).astype(np.uint8)
            
            # Преобразуем в grayscale если нужно
            if len(img_processed.shape) == 3 and img_processed.shape[2] == 3:
                img_processed = np.dot(img_processed[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Нормализуем
            img_processed = img_processed / 255.0
            img_processed = (img_processed - 0.1307) / 0.3081
            
            # Добавляем канал и batch dimension
            img_tensor = torch.FloatTensor(img_processed).unsqueeze(0).unsqueeze(0)
            batch_images.append(img_tensor)
        
        batch = torch.cat(batch_images, dim=0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def _preprocess_image(self, image):
        """Преобразует изображение из LIME формата в тензор"""
        # LIME дает изображения в формате [0,1] uint8
        if image.max() > 1:
            image = image / 255.0
        
        # Для MNIST нужно добавить канал
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        
        # Транспонируем в формат (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Нормализация как при обучении
        mean = 0.1307
        std = 0.3081
        image = (image - mean) / std
        
        return torch.FloatTensor(image)
    
    def explain(self, image, top_labels=5, num_samples=1000):
        """Генерирует LIME объяснение для изображения"""
        self.model.eval()
        
        # Конвертируем тензор в numpy для LIME
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().squeeze()
            image = (image * 0.3081 + 0.1307)  # Денормализация
            image = np.clip(image, 0, 1) #новая строка
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            # Если уже numpy, убедимся что в правильном диапазоне
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)

        # LIME требует RGB изображение, преобразуем grayscale в RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)  # Преобразуем в RGB
        
        from skimage.segmentation import slic
        
        # Получаем объяснение
        explanation = self.explainer.explain_instance(
            image, 
            self.predict_proba,
            top_labels=top_labels,
            num_samples=num_samples,
            hide_color=0,
            num_features=8,
            segmentation_fn=lambda img: slic(img, n_segments=10, compactness=0.1)
        )
        
        return explanation
    
    def get_heatmap(self, explanation, label):
        """Извлекает тепловую карту из объяснения"""
        # Получаем маску важности
        _, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=15,
            hide_rest=False,
            min_weight=0.0  # Игнорируем сегменты с весом меньше 0.05 Новая строка
        )
        
        return mask