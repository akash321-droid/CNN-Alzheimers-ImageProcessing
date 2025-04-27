import torch
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
from model import CombinedCNN_small
from config import img_resize_x, img_resize_y, device, num_classes, class_names
from gradcam import generate_gradcam_visualization

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')

# Hardcoded normalization parameters(pre-calculated in train.py)
DATASET_MEAN = [0.29610347747802734, 0.2961496114730835, 0.2961122393608093]
DATASET_STD = [0.31761598587036133, 0.3176574110984802, 0.31762927770614624]

def load_model(model_path='checkpoints/final_weights.pth'):
    """
    Load the trained model
    
    Args:
        model_path: Path to the saved model weights
        
    Returns:
        model: Loaded model
    """
    model_path = os.path.join(BASE_DIR, model_path)
    model = CombinedCNN_small(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tensor: Preprocessed image tensor
    """
    # Define transforms for test data using hardcoded normalization values
    transform = transforms.Compose([
        transforms.Resize((img_resize_x, img_resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_image(model, image_path):
    """
    Make a prediction for a single image
    
    Args:
        model: Loaded model
        image_path: Path to the image file
        
    Returns:
        predicted_class: Predicted class index
        confidence: Confidence score (probability)
        class_name: Name of the predicted class
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get class name
    class_name = class_names[predicted_class]
    
    return predicted_class, confidence, class_name

def predict_batch(model, image_paths):
    """
    Make predictions for a batch of images
    
    Args:
        model: Loaded model
        image_paths: List of paths to the image files
        
    Returns:
        results: List of dictionaries containing prediction results
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    results = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            results.append({
                'image_path': image_path,
                'error': 'File not found'
            })
            continue
            
        try:
            # Make prediction
            predicted_class, confidence, class_name = predict_image(model, image_path)
            
            # Store results
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'class_name': class_name,
                'confidence': confidence
            })
        except Exception as e:
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results

def analyze_alzheimers_mri(image_paths=None):
    """
    Main prediction function to be imported in interface.py
    
    Args:
        image_paths: List of paths to the image files. If None, processes all images in the data directory.
        
    Returns:
        results: List of dictionaries containing prediction results
    """
    # Load model
    model = load_model()
    
    # If no specific images provided, use all from data directory
    if image_paths is None:
        image_paths = []
        # Look for images directly in the data directory
        for ext in ['jpg', 'jpeg', 'png']:
            image_paths.extend(glob.glob(os.path.join(DEFAULT_DATA_DIR, f'*.{ext}')))
        
        if not image_paths:
            print(f"No images found in {DEFAULT_DATA_DIR}")
            return []
    
    # Make predictions
    results = predict_batch(model, image_paths)
    
    # Print results
    print("\nAlzheimer's MRI Analysis Results:")
    print("--------------------------------")
    
    for result in results:
        if 'error' in result:
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"Error: {result['error']}")
        else:
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"Prediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2%}")
        print("--------------------------------")
    
    return results

def generate_gradcam_for_test_images():
    """
    Generate Grad-CAM visualizations for all test images
    
    Returns:
        None
    """
    output_dir = os.path.join(BASE_DIR, 'gradcam_results')
    
    # Load model
    model = load_model()
    
    # Get target layer for Grad-CAM (last convolutional layer of CNN1)
    # For the CombinedCNN_small model, we'll use the last conv layer of CNN1
    target_layer = model.cnn1.conv3_1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image paths - look for images directly in the data directory
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(DEFAULT_DATA_DIR, f'*.{ext}')))
    
    if not image_paths:
        print(f"No images found in {DEFAULT_DATA_DIR}")
        return
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        # Generate file name for saving
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"gradcam_{file_name}.png")
        
        # Generate and save Grad-CAM visualization
        try:
            _, class_idx, confidence = generate_gradcam_visualization(
                model, image_path, target_layer, save_path=save_path
            )
            print(f"Saved Grad-CAM for {file_name} (Class: {class_names[class_idx]}, Confidence: {confidence:.2%})")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    print(f"Grad-CAM visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alzheimer's MRI Analysis with Grad-CAM")
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'gradcam'],
                        help='Operation mode: predict or gradcam')
    parser.add_argument('--input', type=str, nargs='+', default=None,
                        help='Input image path(s) for prediction mode. If not provided, will use all images in data folder.')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        analyze_alzheimers_mri(args.input)
    
    elif args.mode == 'gradcam':
        generate_gradcam_for_test_images()