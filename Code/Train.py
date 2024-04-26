############### import package #############
import torch
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
############### import package #############



########### quote: https://colab.research.google.com/drive/134NRf4PAR1CuQejf6q3DTDZNaFklR8NY?usp=sharing ############
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
########### quote: https://colab.research.google.com/drive/134NRf4PAR1CuQejf6q3DTDZNaFklR8NY?usp=sharing ############



############### File Path #############
Audio_train = '~/PhD/CS529/CS_529_NN/npy/Apr9/train_mfcc_fold8_50_delta_delta.npy'
Audio_val = '~/PhD/CS529/CS_529_NN/npy/Apr9/val_mfcc_fold8_50_delta_delta.npy'
############### File Path #############


os.path.dirname('Train.py')

############## load ############
Features_Labels = np.load(Audio_train)
#test_Features = np.load('data/npy/Apr9/test_mfcc13_delta_delta.npy')
Val_Features_Labels = np.load(Audio_val)
############## load ############


############### Normalization ###########
mean = np.mean(Features, axis=0, keepdims=True)
std = np.std(Features, axis=0, ddof=1, keepdims=True)

# Z-normalize each row
Features_normalized = (Features - mean) / std

# mean = np.mean(test_Features, axis=1, keepdims=True)
# std = np.std(test_Features, axis=1, ddof=1, keepdims=True)

# Z-normalize each row
test_Features_norm = (test_Features - mean) / std
test_Features_norm

Val_Features_norm = (Val_Features - mean) / std
Val_Features_norm
############### Normalization ###########

X, y = wine.data, wine.target
scaler = StandardScaler()
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = scaler.fit_transform(X_train)
X_test_val = scaler.transform(X_test_val)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize MLP model
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))  # Number of classes

model = MLP(input_size, hidden_size, output_size)
print(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.cost(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    train_accuracy = correct_train / total_train
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    val_accuracy = correct_val / total_val
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%")
