# %%
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from config import img_resize_x, img_resize_y, device, class_names

class GradCAM:
    """
    Grad-CAM implementation for CNN-based models
    
    Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes
    where the model is looking to make its decision
    """
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute Grad-CAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        # Put model in eval mode
        self.model.eval()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        # Save hooks for later removal
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM for the given input tensor
        
        Args:
            input_tensor: Input tensor (should be preprocessed already)
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            cam: Grad-CAM heatmap (numpy array)
            class_idx: Class index used for generating the Grad-CAM
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class if target_class is not specified
        if target_class is None:
            class_idx = output.argmax(dim=1).item()
        else:
            class_idx = target_class
        
        # Clear gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to avoid negative values
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - cam.min()
        if cam.max() > 0:  # Avoid division by zero
            cam = cam / cam.max()
        
        # Resize to input size
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), 
                          mode='bilinear', align_corners=False)
        
        # Convert to numpy array
        cam = cam[0, 0].cpu().numpy()
        
        return cam, class_idx

def preprocess_image(image_path, transform=None):
    """
    Preprocess an image for Grad-CAM
    
    Args:
        image_path: Path to the image file
        transform: Transform to apply (if None, use default)
        
    Returns:
        original_image: Original PIL image
        input_tensor: Preprocessed tensor for model
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    # Use default transform if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_resize_x, img_resize_y)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Apply transform and add batch dimension
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    return original_image, input_tensor

def generate_gradcam_visualization(model, image_path, target_layer, target_class=None, save_path=None):
    """
    Generate and visualize Grad-CAM for an image
    
    Args:
        model: PyTorch model
        image_path: Path to the image file
        target_layer: Layer to compute Grad-CAM for
        target_class: Target class index (if None, use predicted class)
        save_path: Path to save the visualization (if None, do not save)
        
    Returns:
        visualization: Grad-CAM visualization as a numpy array
        class_idx: Class index used for generating the Grad-CAM
        confidence: Confidence score for the prediction
    """
    # Preprocess image
    original_image, input_tensor = preprocess_image(image_path)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
    # Get predicted class and confidence if target_class is not specified
    if target_class is None:
        class_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, class_idx].item()
    else:
        class_idx = target_class
        confidence = probabilities[0, class_idx].item()
        
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, _ = grad_cam.generate_cam(input_tensor, target_class=class_idx)
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    # Convert PIL image to numpy array
    rgb_img = np.array(original_image.resize((img_resize_x, img_resize_y)))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    superimposed = heatmap * 0.4 + rgb_img * 0.6
    superimposed = np.uint8(superimposed)
    
    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axs[0].imshow(rgb_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Heatmap
    axs[1].imshow(heatmap)
    axs[1].set_title('Grad-CAM Heatmap')
    axs[1].axis('off')
    
    # Superimposed
    axs[2].imshow(superimposed)
    class_name = class_names.get(class_idx, f"Class {class_idx}")
    axs[2].set_title(f'Superimposed: {class_name} ({confidence:.2%})')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return superimposed, class_idx, confidence

def compute_average_gradcam_per_class(model, image_paths_by_class, target_layer, output_dir='gradcam_results'):
    """
    Compute average Grad-CAM for each class
    
    Args:
        model: PyTorch model
        image_paths_by_class: Dictionary mapping class indices to lists of image paths
        target_layer: Layer to compute Grad-CAM for
        output_dir: Directory to save results
        
    Returns:
        avg_cams_by_class: Dictionary mapping class indices to average CAMs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Dictionary to store CAMs by class
    cams_by_class = {class_idx: [] for class_idx in image_paths_by_class}
    
    # Generate CAMs for each image
    for class_idx, image_paths in image_paths_by_class.items():
        print(f"Processing class {class_idx} ({class_names.get(class_idx, 'Unknown')})")
        
        for i, image_path in enumerate(image_paths):
            print(f"  Image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Preprocess image
            _, input_tensor = preprocess_image(image_path)
            
            # Generate CAM
            cam, _ = grad_cam.generate_cam(input_tensor, target_class=class_idx)
            
            # Store CAM
            cams_by_class[class_idx].append(cam)
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    # Compute average CAM for each class
    avg_cams_by_class = {}
    for class_idx, cams in cams_by_class.items():
        if cams:  # Check if list is not empty
            avg_cam = np.stack(cams).mean(axis=0)
            avg_cams_by_class[class_idx] = avg_cam
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * avg_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Save heatmap
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            heatmap_path = os.path.join(output_dir, f"avg_gradcam_class_{class_idx}_{class_name}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap)
            plt.title(f'Average Grad-CAM: {class_name}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            
            print(f"Saved average Grad-CAM for class {class_idx} ({class_name}) to {heatmap_path}")
    
    # Create a combined visualization
    plt.figure(figsize=(16, 4))
    for i, (class_idx, avg_cam) in enumerate(avg_cams_by_class.items()):
        plt.subplot(1, len(avg_cams_by_class), i+1)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * avg_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        plt.imshow(heatmap)
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        plt.title(f'{class_name}')
        plt.axis('off')
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "combined_avg_gradcams.png")
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined average Grad-CAMs to {combined_path}")
    
    return avg_cams_by_class


