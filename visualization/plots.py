import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from .utils_viz import denormalize, overlay_lime, overlay_heatmap

def plot_training_curves(train_losses, test_losses, test_accuracies, save_path=None):
    """Строит графики обучения с тестовыми метриками"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График потерь (train и test)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, test_losses, label='Test Loss', color='orange', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График точности
    ax2.plot(epochs, test_accuracies, label='Test Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Строит матрицу ошибок"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_explanations_grid(images, labels, preds, gradcam_heatmaps, lime_masks, 
                           class_names, save_path=None):
    """Строит сетку сравнения объяснений"""
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3*num_images))
    
    # Если только одно изображение, axes нужно сделать двумерным
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Оригинальное изображение
        img = denormalize(images[i]).squeeze()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original\nTrue: {class_names[labels[i]]}')
        axes[i, 0].axis('off')
        
        # Grad-CAM наложение
        gradcam_overlay = overlay_heatmap(images[i], gradcam_heatmaps[i])
        axes[i, 1].imshow(gradcam_overlay)
        pred_color = 'green' if preds[i] == labels[i] else 'red'
        axes[i, 1].set_title(f'Grad-CAM\nPred: {class_names[preds[i]]}', 
                            color=pred_color)
        axes[i, 1].axis('off')
        
        # LIME наложение
        lime_overlay = overlay_lime(images[i], lime_masks[i])
        axes[i, 2].imshow(lime_overlay)
        axes[i, 2].set_title(f'LIME\nPred: {class_names[preds[i]]}', 
                            color=pred_color)
        axes[i, 2].axis('off')
        
        # Сравнение карт важности
        axes[i, 3].imshow(gradcam_heatmaps[i], cmap='jet')
        axes[i, 3].set_title('Heatmap Comparison\n(Grad-CAM)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_tsne(features, labels, class_names, save_path=None):
    """Строит t-SNE визуализацию эмбеддингов"""
    # Уменьшаем размерность
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Рисуем
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_error_analysis(error_data, class_names, save_path=None):
    """Визуализирует ошибочные предсказания с уверенностью"""
    num_errors = len(error_data['images'])
    # Автоматически определяем сетку
    rows = 2
    cols = (num_errors + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 6))
    axes = axes.flatten()
    
    for i in range(num_errors):
        # Изображение
        img = denormalize(error_data['images'][i]).squeeze()
        axes[i].imshow(img, cmap='gray')
        
        # Заголовок с информацией об ошибке
        true_label = error_data['labels'][i].item() if torch.is_tensor(error_data['labels'][i]) else error_data['labels'][i]
        pred_label = error_data['predictions'][i].item() if torch.is_tensor(error_data['predictions'][i]) else error_data['predictions'][i]
        conf = error_data['confidence'][i].item() if torch.is_tensor(error_data['confidence'][i]) else error_data['confidence'][i]
        
        axes[i].set_title(
            f'True: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_label]}\n'
            f'Conf: {conf:.3f}',
            color='red'
        )
        axes[i].axis('off')
    
    # Скрываем лишние подграфики
    for i in range(num_errors, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Typical Classification Errors', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()