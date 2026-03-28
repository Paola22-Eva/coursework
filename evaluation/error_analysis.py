import torch
import numpy as np

def find_correct_and_errors(model, test_loader, device, num_samples=10):
    """Находит правильные и ошибочные предсказания"""
    model.eval()
    correct_images = []
    correct_preds = []
    correct_labels = []
    
    error_images = []
    error_preds = []
    error_labels = []
    error_probs = []  # Уверенность в ошибочном предсказании
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            # Для каждого изображения в батче
            for i in range(len(data)):
                if preds[i] == target[i]:
                    # Правильное предсказание
                    if len(correct_images) < num_samples:
                        correct_images.append(data[i].cpu())
                        correct_preds.append(preds[i].cpu())
                        correct_labels.append(target[i].cpu())
                else:
                    # Ошибочное предсказание
                    if len(error_images) < num_samples:
                        error_images.append(data[i].cpu())
                        error_preds.append(preds[i].cpu())
                        error_labels.append(target[i].cpu())
                        # Уверенность в ошибочном классе
                        error_probs.append(probs[i][preds[i]].cpu())
            
            # Останавливаемся, если набрали достаточно
            if len(correct_images) >= num_samples and len(error_images) >= num_samples:
                break
    
    return {
        'correct': {
            'images': correct_images[:num_samples],
            'predictions': correct_preds[:num_samples],
            'labels': correct_labels[:num_samples]
        },
        'error': {
            'images': error_images[:num_samples],
            'predictions': error_preds[:num_samples],
            'labels': error_labels[:num_samples],
            'confidence': error_probs[:num_samples]
        }
    }

def analyze_errors(error_data, class_names):
    """Анализирует типичные ошибки модели"""
    print("Анализ ошибок классификации:")
    print("-" * 50)
    
    # Считаем самые частые ошибки
    error_pairs = []
    for true, pred in zip(error_data['labels'], error_data['predictions']):
        error_pairs.append((true.item(), pred.item()))
    
    from collections import Counter
    common_errors = Counter(error_pairs).most_common(5)
    
    print("Самые частые ошибки:")
    for (true, pred), count in common_errors:
        print(f"  {class_names[true]} -> {class_names[pred]}: {count} раз")
    
    # Анализ уверенности в ошибках
    if error_data['confidence']:
        avg_confidence = np.mean([c.item() for c in error_data['confidence']])
        print(f"\nСредняя уверенность при ошибках: {avg_confidence:.3f}")