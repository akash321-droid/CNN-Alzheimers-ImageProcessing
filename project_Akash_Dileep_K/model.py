import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the SCALED DOWN first CNN branch (CNN1_small)
class CNN1_small(nn.Module):
    def __init__(self):
        super(CNN1_small, self).__init__()
        # Reduced Convolutional layers
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same') # Reduced from 16
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same') # Reduced from 16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same') # Reduced from 64
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')# Reduced from 64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding='same')# Reduced from 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch Normalization and Dropout
        self.batchnorm = nn.BatchNorm2d(64) # Adjusted channel count
        self.dropout = nn.Dropout(0.2) # Keeping dropout rate

        # Reduced Fully Connected Layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) # Keep pooling consistent
        self.flattened_size = 64 * 7 * 7 # Adjusted channel count
        self.fc1 = nn.Linear(self.flattened_size, 64) # Reduced from 128

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = self.pool3(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

# Define the SCALED DOWN second CNN branch (CNN2_small)
class CNN2_small(nn.Module):
    def __init__(self):
        super(CNN2_small, self).__init__()
        # Reduced Convolutional layers
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding='same') # Reduced from 32
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding='same') # Reduced from 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding='same') # Reduced from 128
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same')# Reduced from 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, padding='same')# Reduced from 512
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch Normalization and Dropout
        self.batchnorm = nn.BatchNorm2d(128) # Adjusted channel count
        self.dropout = nn.Dropout(0.2) # Keeping dropout rate

        # Reduced Fully Connected Layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) # Keep pooling consistent
        self.flattened_size = 128 * 7 * 7 # Adjusted channel count
        self.fc1 = nn.Linear(self.flattened_size, 64) # Reduced from 128

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = self.pool3(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

# Define the SCALED DOWN combined model
class CombinedCNN_small(nn.Module):
    def __init__(self, num_classes=4): # Set default to 4 classes for your dataset
        super(CombinedCNN_small, self).__init__()
        self.cnn1 = CNN1_small()
        self.cnn2 = CNN2_small()

        # Concatenation layer and final classification layer
        # Input size is 64 (from CNN1_small) + 64 (from CNN2_small) = 128
        self.fc_concat = nn.Linear(64 + 64, num_classes)

    def forward(self, x):
        out1 = self.cnn1(x)
        out2 = self.cnn2(x)
        concat_features = torch.cat((out1, out2), dim=1)
        output = self.fc_concat(concat_features)
        # Remember: Apply Softmax *after* this if not using nn.CrossEntropyLoss
        return output