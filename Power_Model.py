# Power consumption single-node machine learning model
# Used RandomForestClassifier

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('power_consumption_data.csv')

# Encode 'Subkey' and 'Attack/Normal' into numerical values
le = LabelEncoder()
df['Subkey'] = le.fit_transform(df['Subkey'])  # Active = 1, Idle = 0
df['Attack/Normal'] = df['Attack/Normal'].map({"Attack": 1, "Normal": 0})

# Features and labels
X = df[['Power Trace (Volts)', 'Subkey']]
y = df['Attack/Normal']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start time
start_time = time.time()

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
# Plot Confusion Matrix
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Extracting overall success rate
correct_predictions = sum(y_pred == y_test)
total_predictions = len(y_test)
success_rate = correct_predictions / total_predictions

# Print results
print(f"Training Time: {training_time:.4f} seconds")
print(f"Success Rate: {success_rate * 100:.2f}%")





