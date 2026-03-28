import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """Реализация Grad-CAM"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
    
                # Регистрируем хуки
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    
    def save_activation(self, module, input, output):
        """Сохраняет активации при forward проходе"""
        self.activations = output
        print(f"  Activation captured - shape: {self.activations.shape}")
    
    def save_gradient(self, module, grad_input, grad_output):
        """Сохраняет градиенты при backward проходе"""
        self.gradients = grad_output[0]
        print(f"  Gradient captured - shape: {self.gradients.shape}")

    
    def generate_heatmap(self, input_image, target_class=None):
        """Генерирует тепловую карту для изображения"""
        self.model.eval()

        print(f"\n--- Generating heatmap ---")
        print(f"Input image shape: {input_image.shape}")
        
        # Forward pass
        output = self.model(input_image.unsqueeze(0))
        print(f"Output shape: {output.shape}")
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.last_pred = target_class	
        print(f"Target class: {target_class}")
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        # Проверяем, что хуки сработали
        if self.gradients is None:
            raise ValueError("Gradients not captured! Check hook registration.")
        if self.activations is None:
            raise ValueError("Activations not captured! Check hook registration.")
        
        print(f"Gradients shape before processing: {self.gradients.shape}")
        print(f"Activations shape before processing: {self.activations.shape}")
        
        # Получаем градиенты и активации
        gradients = self.gradients.cpu().data.numpy()[0].copy()
        activations = self.activations.cpu().data.numpy()[0].copy()

        # Считаем веса каналов (Global Average Pooling градиентов)
        weights = np.mean(gradients, axis=(1, 2))

        
        # Взвешенная сумма карт активации
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU и нормализация
        #cam = np.maximum(cam, 0)

        # Изменяем размер до 28x28
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[1]), 
                         interpolation=cv2.INTER_LINEAR)
        
        # Нормализация
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min()+ 1e-8)
        else:
            cam = np.zeros_like(cam)

        print(f"Normalized CAM - min: {cam.min():.4f}, max: {cam.max():.4f}")
        
        return cam