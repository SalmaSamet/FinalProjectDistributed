#Code is in Distributed setup within PyTorch using DistributedDataParallel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time



import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

# Dataset class to load data from CSV
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # Load data from CSV using pandas
        data_frame = pd.read_csv(csv_file)
        
        # Assuming last column is the label
        self.data = torch.tensor(data_frame.iloc[:, :-1].values, dtype=torch.float32)  # Features
        self.labels = torch.tensor(data_frame.iloc[:, -1].values, dtype=torch.long)   # Labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Your neural network model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 10)  # Assuming image size 32x32

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def setup(rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU, 'nccl' for GPU
        init_method='env://',  # Initializes using environment variables
        world_size=world_size,  # Total number of processes
        rank=rank  # Rank of the current process
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create a model and move it to the corresponding GPU (if using GPUs)
    model = SimpleCNN().to(rank)
    
    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    # Assuming dataset is loaded from CSV, update here accordingly
    # Example: data = torch.tensor(...), labels = torch.tensor(...)
    # For now, using random data as a placeholder
    data = torch.randn(100, 3, 32, 32)  # Example data, 100 images of 32x32 RGB
    labels = torch.randint(0, 10, (100,))  # Example labels
    dataset = SimpleDataset(data, labels)

    # Use DistributedSampler to partition the data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )

    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Variables to track loss, accuracy, and runtime
    total_loss = 0.0
    total_accuracy = 0.0
    total_time = 0.0
    num_batches = 0

    # Training loop
    for epoch in range(10):
        start_time = time.time()  # Track the time for this epoch
        sampler.set_epoch(epoch)  # Important to shuffle the data differently at each epoch

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(rank)  # Move inputs to the correct device
            targets = targets.to(rank)  # Move targets to the correct device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()

            num_batches += 1

        # End of epoch
        epoch_end_time = time.time()  # Time after epoch
        epoch_runtime = epoch_end_time - start_time
        total_time += epoch_runtime

        # Calculate the average loss and accuracy for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_accuracy = 100.0 * epoch_correct / epoch_total

        total_loss += avg_epoch_loss
        total_accuracy += epoch_accuracy

        # Print out stats for this epoch
        print(f"Rank {rank}, Epoch [{epoch+1}/10], "
              f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
              f"Time: {epoch_runtime:.2f} seconds")

    # After the training loop
    avg_loss = total_loss / 10  # Average over all epochs
    avg_accuracy = total_accuracy / 10  # Average over all epochs
    avg_runtime = total_time / 10  # Average runtime per epoch

    # Print final results
    print(f"\nRank {rank} - Final Results: "
          f"Avg. Loss: {avg_loss:.4f}, Avg. Accuracy: {avg_accuracy:.2f}%, "
          f"Avg. Runtime: {avg_runtime:.2f} seconds")

    cleanup()

if __name__ == "__main__":
    world_size = 2  # Number of processes to run (adjust as needed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    csv_file = 'allcombined.csv'
    
    # Launch multiple processes for data parallelism
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

