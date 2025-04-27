# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CombinedCNN_small
from dataset import prepare_data_loaders
from config import (
    learning_rate, weight_decay, epochs, 
    early_stopping_patience, data_dir, device
)

def train_model(
    model, 
    epochs, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device,
    scheduler=None,
    early_stopping_patience=None
):
    """
    Train the model using the provided data loaders
    
    Args:
        model: PyTorch model to train
        epochs: Number of epochs to train for
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cpu or cuda)
        scheduler: Learning rate scheduler (optional)
        early_stopping_patience: Number of epochs to wait for improvement (optional)
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f'Saving best model with validation loss: {best_val_loss:.4f}')
                torch.save(model.state_dict(), 'checkpoints/final_weights.pth')
            else:
                patience_counter += 1
                print(f'Early stopping counter: {patience_counter}/{early_stopping_patience}')
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break
    
    # If not using early stopping, save the final model
    if early_stopping_patience is None:
        torch.save(model.state_dict(), 'checkpoints/final_weights.pth')
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return model, history

def plot_training_history(history):
    """Plot the training and validation loss/accuracy curves"""
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def run_training():
    """Main function to run the training process"""
    # Set device
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        test_size=0.2,
        val_size=0.2
    )
    
    # Initialize model
    model = CombinedCNN_small(num_classes=4)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,)
    
    # Train the model
    model, history = train_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model

def evaluate_model(model, data_loader, device):
    """Evaluate the model on a given dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    run_training()


