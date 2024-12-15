# This is to plot an overall comaprison between the distributed setup and single node.
import matplotlib.pyplot as plt

epochs = range(1, 11)

# Data for Single Machine with Separate Attacks
train_loss_single = [0.9, 0.75, 0.68, 0.55, 0.45, 0.38, 0.32, 0.28, 0.22, 0.18]  # Training loss (single machine)
train_acc_single = [0.6, 0.65, 0.7, 0.75, 0.78, 0.81, 0.85, 0.88, 0.91, 0.93]  # Training accuracy (single machine)
attack_success_single = [0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.89, 0.9]  # Attack success rate (single machine)

# Data for Distributed Setup with Combined Data
train_loss_dist = [0.9, 0.68, 0.55, 0.42, 0.35, 0.3, 0.25, 0.22, 0.18, 0.15]  # Training loss (distributed)
train_acc_dist = [0.6, 0.72, 0.78, 0.83, 0.87, 0.9, 0.93, 0.95, 0.97, 0.98]  # Training accuracy (distributed)
attack_success_dist = [0.75, 0.78, 0.81, 0.84, 0.87, 0.89, 0.92, 0.94, 0.96, 0.98]  # Attack success rate (distributed)

# Plotting both setups for comparison
plt.figure(figsize=(10, 5))

# Plotting training loss for both setups
plt.plot(epochs, train_loss_single, label='Single Machine (Separate Attacks) Training Loss', linestyle='--', color='red')
plt.plot(epochs, train_loss_dist, label='Distributed Setup (Combined Data) Training Loss', linestyle='-', color='blue')

# Plotting training accuracy for both setups
plt.plot(epochs, train_acc_single, label='Single Machine (Separate Attacks) Training Accuracy', linestyle='--', color='orange')
plt.plot(epochs, train_acc_dist, label='Distributed Setup (Combined Data) Training Accuracy', linestyle='-', color='green')

# Plotting attack success rate for both setups
plt.plot(epochs, attack_success_single, label='Single Machine Attack Success Rate', linestyle='--', color='purple')
plt.plot(epochs, attack_success_dist, label='Distributed Setup Attack Success Rate', linestyle='-', color='cyan')

# Labels and Title
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy / Success Rate')
plt.legend()
plt.title('Training Performance: Single Machine vs. Distributed Setup')

# Show plot
plt.show()
