import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def denormalize(tensor, mean=0.1307, std=0.3081):
    """Денормализует изображение для визуализации"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().clone()
        # Денормализация: x = x * std + mean
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.numpy()
    else:
        return tensor

def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
   # Накладывает тепловую карту на изображение
    # Конвертируем в numpy
    if isinstance(image, torch.Tensor):
        image = denormalize(image).squeeze()
    
    # Нормализуем для отображения
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    
    # Конвертируем в 3-канальное
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Применяем цветовую карту
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Накладываем
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay

def overlay_lime(image, mask, alpha=0.5):
    """Накладывает LIME маску на изображение, выделяя важные сегменты зеленым"""
    # Получаем денормализованное изображение
    if isinstance(image, torch.Tensor):
        img_np = denormalize(image).squeeze()
    else:
        img_np = image.copy()
    
    # Нормализуем
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Конвертируем в RGB
    if len(img_np.shape) == 2:
        img_rgb = np.stack([img_np, img_np, img_np], axis=2)
    else:
        img_rgb = img_np
    
    # Создаем зеленую маску для важных сегментов (mask содержит номера сегментов > 0)
    colored_mask = np.zeros_like(img_rgb)
    colored_mask[mask > 0] = [0, 1, 0]  # Ярко-зеленый
    
    # Накладываем маску
    overlay = img_rgb * (1 - alpha) + colored_mask * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay