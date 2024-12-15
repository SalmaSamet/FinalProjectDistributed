#Timing attack single-node machine learning model
#Used 
#Binary classification ( 0 = no attack, 1= attack detected)


import pandas as pd
import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Timing_attack_dataset.csv')

# Features and Labels
X = data[['Execution_Time']].values
y = data['Key_Bit'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)    
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the LSTM Model
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # Take the output of the last hidden state
        return out

# Model initialization
input_size = 1  # One feature: Execution_Time
hidden_size = 64
num_classes = 2  # Binary classification
model = LSTM_Model(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store loss and accuracy for each epoch
losses = []
accuracies = []
test_accuracies = []

# Training the model
start_time = time.time()
for epoch in range(100):  # Number of epochs
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    _, predicted = torch.max(output, 1)
    train_accuracy = accuracy_score(y_train_tensor.cpu().numpy(), predicted.cpu().numpy())
    
    losses.append(loss.item())
    accuracies.append(train_accuracy)

    # Calculate test accuracy
    model.eval()
    with torch.no_grad():
        output_test = model(X_test_tensor)
        _, predicted_test = torch.max(output_test, 1)
        test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy())
        
    test_accuracies.append(test_accuracy)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%")

# Plotting the Loss and Accuracy curves
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, 101), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 101), accuracies, label='Training Accuracy', color='orange')
plt.plot(range(1, 101), test_accuracies, label='Testing Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Testing the model
model.eval()
with torch.no_grad():
    output = model(X_test_tensor)
    _, predicted = torch.max(output, 1)
    
    # Iterate through the batch and print predicted labels
    for label in predicted:
        attack_label = 'Attack' if label.item() == 1 else 'No Attack'
        print(f"Predicted: {attack_label}")
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())

print(f"Test Accuracy: {test_accuracy*100:.2f}%")


# Print results
end_time = time.time()
training_time = end_time - start_time
attack_success_rate = test_accuracy * 100  
loss_rate = loss.item()

print(f"Training Time: {training_time:.2f} seconds")
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
print(f"Attack Success Rate: {attack_success_rate:.2f}%")
print(f"Final Loss Rate: {loss_rate:.4f}")



