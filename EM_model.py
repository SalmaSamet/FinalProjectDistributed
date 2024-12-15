#CNN classification machine learning model for EM attack

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the CSV file
data = pd.read_csv('em_data1.csv')

# Step 2: Handle missing data (imputation)
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data.drop(columns=['label', 'attack_success']))

# Rebuild the DataFrame after imputation
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[:-2])

# Add back the target columns (label and attack_success)
data_imputed['label'] = data['label']
data_imputed['attack_success'] = data['attack_success']

# Step 3: Split the data into features (X) and target (y)
X = data_imputed.drop(columns=['label', 'attack_success'])
y = data_imputed['attack_success']

# Step 4: Reshape the data to match CNN input format
X = X.values.reshape(X.shape[0], X.shape[1], 1)  # Shape: (num_samples, time_length, 1)

# Convert the target variable to categorical (for classification)
y = np.array(y)

# Step 5: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 6: Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (X_train.shape[1] // 4), 64)  # Adjusting the size after convolution layers
        self.fc2 = nn.Linear(64, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 7: Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Train the model and track time
start_time = time.time()

# Create data loaders
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.unsqueeze(1))  # Add channel dimension (1D input)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

end_time = time.time()

# Step 9: Evaluate the model on the test data
model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.unsqueeze(1))  # Add channel dimension (1D input)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)

# Calculate training time
training_time = end_time - start_time

# Print final results
print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

