#####config.py#####


# %%
import torch
import os

# Model configuration
num_classes = 4  # Number of Alzheimer's classification classes

# Image preprocessing
img_resize_x = 224
img_resize_y = 224
input_channels = 3  # RGB images

# Training hyperparameters
batch_size = 32
epochs = 50
learning_rate = 0.001
weight_decay = 0.001
early_stopping_patience = 10

# Dataset configuration
data_dir = "data"
test_size = 0.2
val_size = 0.2
random_state = 42

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Directories
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# Class names
class_names = {
    0: "non_demented",
    1: "very_mild_dementia", 
    2: "mild_dementia",
    3: "moderate_dementia"
}


