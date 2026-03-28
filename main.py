"""
Основной скрипт для курсовой работы "Объяснимость модели (Grad-CAM / LIME)"
Запускает полный пайплайн: обучение -> оценка -> объяснения -> визуализация
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Добавляем пути для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наши модули
from config.config import Config, set_seed
from data.data_loader import get_data_loaders, get_sample_images
from models.model import CNNModel
from training.training import Trainer
from evaluation.metrics import calculate_metrics
from evaluation.error_analysis import find_correct_and_errors, analyze_errors
from explainability.gradcam import GradCAM
from explainability.lime_explainer import LIMEExplainer
from visualization.plots import (
    plot_training_curves, 
    plot_confusion_matrix, 
    plot_explanations_grid,
    plot_tsne,
    plot_error_analysis
)

def print_step(step_num, description):
    """Выводит шаг выполнения в терминал"""
    print(f"\n{'='*60}")
    print(f"ШАГ {step_num}: {description}")
    print(f"{'='*60}")

def main():
    """Главная функция пайплайна"""
    
    # ==================== ШАГ 1: ИНИЦИАЛИЗАЦИЯ ====================
    print_step(1, "Инициализация конфигурации и фиксация seed'ов")
    
    # Создаем конфигурацию
    config = Config()
    
    # Фиксируем seed для воспроизводимости
    set_seed(config.seed)
    print(f" Seed фиксирован: {config.seed}")
    print(f" Устройство: {config.device}")
    print(f" Датасет: {config.dataset}")
    print(f" Batch size: {config.batch_size}")
    print(f" Эпохи: {config.epochs}")
    
    # Создаем папки для сохранения результатов
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, 'logs'), exist_ok=True)
    
    # ==================== ШАГ 2: ЗАГРУЗКА ДАННЫХ ====================
    print_step(2, "Загрузка данных MNIST")
    
    train_loader, test_loader = get_data_loaders(config)
    print(f"Обучающих примеров: {len(train_loader.dataset)}")
    print(f"Тестовых примеров: {len(test_loader.dataset)}")
    
    # Получаем изображения для визуализации
    #sample_images, sample_labels = get_sample_images(test_loader, config.num_test_images)
    #print(f"Отобрано {len(sample_images)} тестовых изображений для анализа")
    
    # ==================== ШАГ 3: СОЗДАНИЕ МОДЕЛИ ====================
    print_step(3, "Создание CNN модели")
    
    model = CNNModel(num_classes=config.num_classes)
    model = model.to(config.device)
    print(f"Модель создана и перенесена на {config.device}")
    
    # ==================== ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ ====================
    print_step(4, "Обучение модели (100 эпох)")
    
    trainer = Trainer(model, train_loader, test_loader, config)
    
    # Проверяем, есть ли уже обученная модель
    checkpoint_path = os.path.join(config.checkpoint_dir, 'model.pth')
    if os.path.exists(checkpoint_path):
        print("Найден сохраненный чекпоинт, загружаем модель...")
        trainer.load_model()
    else:
        print("Начинаем обучение...")
        trainer.train()
        print("Обучение завершено!")
    
    # Сохраняем графики обучения
    plot_training_curves(
        trainer.train_losses,
        trainer.test_losses, 
        trainer.test_accuracies,
        save_path=os.path.join(config.results_dir, 'figures', 'training_curves.png')
    )
    print("Графики обучения сохранены")
    
    # ==================== ШАГ 5: ОЦЕНКА МОДЕЛИ ====================
    print_step(5, "Оценка качества модели")
    
    metrics = calculate_metrics(model, test_loader, config.device)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1-score: {metrics['f1']:.4f}")
    
    # Сохраняем матрицу ошибок
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=os.path.join(config.results_dir, 'figures', 'confusion_matrix.png')
    )
    print("Матрица ошибок сохранена")
    
    # ==================== ШАГ 6: АНАЛИЗ ОШИБОК ====================
    print_step(6, "Анализ типичных ошибок модели")
    
    error_analysis = find_correct_and_errors(
        model, test_loader, config.device, num_samples=10
    )
    
    print(f"Найдено правильных предсказаний: {len(error_analysis['correct']['images'])}")
    print(f"Найдено ошибочных предсказаний: {len(error_analysis['error']['images'])}")
    
    # Анализируем ошибки
    analyze_errors(error_analysis['error'], class_names)
    
    # Визуализируем ошибки
    plot_error_analysis(
        error_analysis['error'],
        class_names,
        save_path=os.path.join(config.results_dir, 'figures', 'error_analysis.png')
    )
    print("Визуализация ошибок сохранена")
    
    # ==================== ШАГ 7: t-SNE ВИЗУАЛИЗАЦИЯ ====================
    print_step(7, "Построение t-SNE визуализации эмбеддингов")
    
    # Извлекаем эмбеддинги для тестовых данных
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(config.device)
            # Получаем эмбеддинги после conv слоев
            _ = model(data)
            features = model.get_features()
            # Global Average Pooling
            emb = features.mean(dim=[2, 3])
            embeddings.append(emb.cpu().numpy())
            labels_list.append(target.numpy())
    
    embeddings = np.concatenate(embeddings)
    labels_list = np.concatenate(labels_list)
    
    # Ограничиваем для t-SNE (берем 1000 точек для скорости)
    indices = np.random.choice(len(embeddings), 1000, replace=False)
    plot_tsne(
        embeddings[indices],
        labels_list[indices],
        class_names,
        save_path=os.path.join(config.results_dir, 'figures', 'tsne.png')
    )
    print("t-SNE визуализация сохранена")
    
# ==================== ШАГ 8: ПОДГОТОВКА НАБОРОВ ====================
    print_step(8, "Подготовка наборов изображений для объяснений")

    # Извлекаем правильные и ошибочные изображения из error_analysis
    correct_imgs = error_analysis['correct']['images']
    correct_labels = error_analysis['correct']['labels']
    error_imgs = error_analysis['error']['images']
    error_labels = error_analysis['error']['labels']

    print(f"Доступно правильных изображений: {len(correct_imgs)}")
    print(f"Доступно ошибочных изображений: {len(error_imgs)}")

    # Берем до 5 правильных и до 5 ошибочных
    num_correct = min(5, len(correct_imgs))
    num_error = min(5, len(error_imgs))

    correct_imgs = correct_imgs[:num_correct]
    correct_labels = correct_labels[:num_correct]
    error_imgs = error_imgs[:num_error]
    error_labels = error_labels[:num_error]

    # ==================== ШАГ 9: GRAD-CAM ДЛЯ ПРАВИЛЬНЫХ ====================
    print_step(9, "Grad-CAM для правильных предсказаний")
    target_layer = model.conv5_block
    gradcam = GradCAM(model, target_layer)

    correct_gradcam_heatmaps = []
    correct_gradcam_preds = []
    model.eval()
    for img in correct_imgs:
        img = img.to(config.device)
        heatmap = gradcam.generate_heatmap(img)
        correct_gradcam_heatmaps.append(heatmap)
        correct_gradcam_preds.append(gradcam.last_pred)
    print(f"Grad-CAM для правильных: {len(correct_gradcam_heatmaps)}")

    # ==================== ШАГ 10: GRAD-CAM ДЛЯ ОШИБОЧНЫХ ====================
    print_step(10, "Grad-CAM для ошибочных предсказаний")
    error_gradcam_heatmaps = []
    error_gradcam_preds = []
    for img in error_imgs:
        img = img.to(config.device)
        heatmap = gradcam.generate_heatmap(img)
        error_gradcam_heatmaps.append(heatmap)
        error_gradcam_preds.append(gradcam.last_pred)
    print(f"✓ Grad-CAM для ошибочных: {len(error_gradcam_heatmaps)}")

    # ==================== ШАГ 11: LIME ДЛЯ ПРАВИЛЬНЫХ ====================
    print_step(11, "LIME для правильных предсказаний")
    lime_explainer = LIMEExplainer(model, config.device)

    correct_lime_masks = []
    correct_lime_preds = []
    for img in correct_imgs:
        img_np = img.cpu().numpy().squeeze()
        img_np = (img_np * 0.3081 + 0.1307)  # денормализация
        explanation = lime_explainer.explain(img_np, num_samples=500)
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(config.device))
            pred = output.argmax(dim=1).item()
        mask = lime_explainer.get_heatmap(explanation, pred)
        correct_lime_masks.append(mask)
        correct_lime_preds.append(pred)
    print(f"LIME для правильных: {len(correct_lime_masks)}")

    # ==================== ШАГ 12: LIME ДЛЯ ОШИБОЧНЫХ ====================
    print_step(12, "LIME для ошибочных предсказаний")
    error_lime_masks = []
    error_lime_preds = []
    for img in error_imgs:
        img_np = img.cpu().numpy().squeeze()
        img_np = (img_np * 0.3081 + 0.1307)
        explanation = lime_explainer.explain(img_np, num_samples=500)
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(config.device))
            pred = output.argmax(dim=1).item()
        mask = lime_explainer.get_heatmap(explanation, pred)
        error_lime_masks.append(mask)
        error_lime_preds.append(pred)
    print(f"LIME для ошибочных: {len(error_lime_masks)}")

    # ==================== ШАГ 13: ВИЗУАЛИЗАЦИЯ ПРАВИЛЬНЫХ ====================
    print_step(13, "Визуализация правильных предсказаний")
    if len(correct_imgs) > 0:
        plot_explanations_grid(
            correct_imgs,
            correct_labels,
            correct_gradcam_preds,
            correct_gradcam_heatmaps,
            correct_lime_masks,
            class_names,
            save_path=os.path.join(config.results_dir, 'figures', 'correct_explanations.png')
        )
        print("Визуализация правильных предсказаний сохранена")
    else:
        print("Нет правильных предсказаний для визуализации")

    # ==================== ШАГ 14: ВИЗУАЛИЗАЦИЯ ОШИБОЧНЫХ ====================
    print_step(14, "Визуализация ошибочных предсказаний")
    if len(error_imgs) > 0:
        plot_explanations_grid(
            error_imgs,
            error_labels,
            error_gradcam_preds,
            error_gradcam_heatmaps,
            error_lime_masks,
            class_names,
            save_path=os.path.join(config.results_dir, 'figures', 'error_explanations.png')
        )
        print("Визуализация ошибочных предсказаний сохранена")
    else:
        print("Нет ошибочных предсказаний для визуализации")

    # ==================== ФИНАЛЬНЫЙ ОТЧЕТ ====================
    print_step(12, "Эксперимент завершен!")
    
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
    print(f"{'='*60}")
    print(f"Точность модели: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Grad-CAM и LIME объяснения сгенерированы")
    print(f"\nВсе результаты сохранены в папку: {config.results_dir}/")
    print(f"  - figures/     : графики и визуализации")
    print(f"  - logs/        : логи эксперимента")
    print(f"  - checkpoints/ : веса модели")
    
    print(f"\n{'='*60}")
    print("Программа закончилась!")
    print(f"{'='*60}")

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"\n Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
