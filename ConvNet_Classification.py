import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

# Define the CNN

'''
100  010  001
010  010  010
001  010  100  其他
 1    2    3    0
'''


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2)  # 1 input channel, 3 output channels
        self.fc1 = nn.Linear(3 * 2 * 2, 4)  # Fully connected layer: 3 feature maps of size 2x2, output 4 classes
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply convolution and ReLU
        x = x.view(-1, 3 * 2 * 2)  # Flatten the output
        x = self.fc1(x)  # Fully connected layer
        return F.log_softmax(x, dim=1)  # Log Softmax for multi-class classification

class ImprovedConvNet(nn.Module):
    def __init__(self):
        super(ImprovedConvNet, self).__init__()
        '''
        Input Channels (in_channels):
            This represents the number of channels in the input data. For a grayscale image, this is 1 (because it has one channel). 
            For an RGB image, it would be 3.

        Output Channels (out_channels):
            This represents the number of filters (or feature maps) the layer will learn. 
            Each filter extracts a different feature from the input. 
            If you set out_channels to 16, the layer will produce 16 different feature maps (each corresponding to a different filter).

        Kernel Size (kernel_size):
            The kernel size is the dimension of the window that slides over the image to apply the convolution operation. 
            A 3x3 kernel means that each filter will look at a 3x3 area of the image at a time.

        '''
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 input channel, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32 output channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 output channels
        
        '''
        The fc size is determined by the previous layer, stride, padding and kernel size together.
        '''
        self.fc1 = nn.Linear(64 * 3 * 3, 64)

        '''
        output is 4 because there are 4 classes
        '''
        self.fc2 = nn.Linear(64, 4)
        
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        '''
        Normalized Output = (X - batch_mean)/(batch_standard_deviation + epsilon)

        - Batch Mean
        The batch mean is the average value of the outputs for a specific feature across all examples in the mini-batch.
        If you have a mini-batch of size 𝑁 and each output has 𝐶 channels (features), then the mean for each channel 
        is computed by averaging the values across the 𝑁 examples.

        - Batch Standard Deviation
        The batch standard deviation measures the spread of the outputs for a specific feature across all examples in the mini-batch.
        It is computed by taking the square root of the variance, which is the average of the squared differences from the mean.

        During training, for each mini-batch, the mean and standard deviation are computed and used to normalize the outputs of the layer. 
        This ensures that the outputs have a mean of 0 and a standard deviation of 1, before being scaled and shifted by the learned parameters 
        𝛾(scale) and 𝛽(shift). 
        
        '''

        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Initialize the model, loss function, and optimizer
model = ImprovedConvNet()

'''
hook for activation capture
'''

# Dictionary to store activations
activations = dict()

# Function to register hook
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on the layers you want to monitor
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))


criterion = nn.CrossEntropyLoss()

# Previous implementation, naive Adam optimizer.
# optimizer = optim.Adam(model.parameters(), lr=0.001)

initial_lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5) 
# Implement a learning rate scheduler that reduces the learning rate as training progresses.

# Learning rate warm-up scheduler
warmup_epochs = 5

# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1)
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

'''
# Sample data: 3x3 binary matrices (as input) and corresponding labels
# Example dataset for simplicity (you should generate a larger one for actual training)
data = torch.tensor([
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 0], [1, 0, 0]], # if this pattern doesn't represent in the training dataset, then it will not recognize the pattern
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Class 1
    [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # Class 2
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]],  # Class 3
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # Class 0
], dtype=torch.float32).unsqueeze(1)  # Add channel dimension

labels = torch.tensor([0, 0, 0, 1, 2, 3, 0], dtype=torch.long)
'''

def generate_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        matrix = torch.randint(0, 2, (3, 3)).float()
        if torch.all(matrix.diag() == 1):
            label = 1  # Top-left to bottom-right diagonal
        elif torch.all(matrix[:, 1] == 1):
            label = 2  # Vertical line in the middle
        elif torch.all(torch.fliplr(matrix).diag() == 1):
            label = 3  # Top-right to bottom-left diagonal
        else:
            label = 0  # Any other pattern
        data.append(matrix)
        labels.append(label)
    return torch.stack(data), torch.tensor(labels)

# Generate dataset
data, labels = generate_data()

# Manually split the data (80% training, 20% validation)
num_samples = data.shape[0]
indices = torch.randperm(num_samples)
train_size = int(0.8 * num_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_data = data[train_indices].unsqueeze(1)
train_labels = labels[train_indices].long()

val_data = data[val_indices].unsqueeze(1)
val_labels = labels[val_indices].long()

#Now we use data generator for training
# Create data loaders
batch_size = 64
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

#Now we add Val dataset used for validation

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# val_data = torch.tensor([
#     [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Class 1
#     [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # Class 2
#     [[0, 0, 1], [0, 1, 0], [1, 0, 0]],  # Class 3
#     [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
#     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # Class 0
    
# ], dtype=torch.float32).unsqueeze(1)  # Add channel dimension

# val_labels = torch.tensor(torch.tensor([0, 1, 2, 3, 0, 0, 0], dtype=torch.long))

train_losses = []
gradients = []
activation_histograms = {}
# Training loop

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loss = 0.0
    running_loss = 0.0
    for batch_data, batch_labels in train_loader: 
        '''
        After processing a batch, the model updates its weights using the gradients computed from that batch.
        This means the model improves incrementally, adjusting its weights multiple times within an epoch.
        '''
        optimizer.zero_grad()
        outputs = model(batch_data)
    
        loss = criterion(outputs, batch_labels)
        loss.backward()

        '''
        Implement gradient clipping to prevent the gradients from becoming too large, 
        which can stabilize training. In PyTorch, this can be done using torch.nn.utils.clip_grad_norm_.
        '''

        # Gradient clipping
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        gradients.append(sum(grad_norms) / len(grad_norms))

        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.item() * batch_data.size(0)

    if len(activation_histograms.keys())  == 0:
        # Initialize lists to store activations histograms per epoch
        activation_histograms = {name: [] for name in activations.keys()}

    # Store activation histograms for this batch
    for name, activation in activations.items():
        activation_histograms[name].append(activation.cpu().numpy())

        
    # Update learning rate
    scheduler.step()

    train_loss /= len(train_loader.dataset)  # Normalizing by the dataset size
    train_losses.append(train_loss)

    '''
    After all batches are processed, the model has gone through the entire dataset, 
    and you can assess how well the model has generalized by calculating metrics like validation accuracy or loss.
    '''
        
    # optimizer.zero_grad()
    # output = model(data)
    # loss = criterion(output, labels)
    # loss.backward()
    # optimizer.step()
    # running_loss += loss.item()
    # train_loss += loss.item() * data.size(0)
    # train_loss /= data.size(0)
    # train_losses.append(train_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        
        # val_outputs = model(val_data)
        wrong_index = []
        
        for val_data_batch, val_labels_batch in val_loader:
            val_outputs = model(val_data_batch)
            val_loss += criterion(val_outputs, val_labels_batch).item()
            _, predicted = torch.max(val_outputs, dim=1)
            correct += (predicted == val_labels_batch).sum().item()
            total += val_labels_batch.size(0)
            # Step 2: Compare predicted classes with ground truth labels
            incorrect_indices = torch.nonzero(predicted != val_labels_batch).squeeze()
            if incorrect_indices.numel() > 0:
                # Step 3: Gather wrong predictions and correct labels
                wrong_predictions = predicted[incorrect_indices]
                correct_labels = val_labels[incorrect_indices]
        
                # if len(wrong_predictions.tolist()) != 0:
                    # Step 4: Output the indices, wrong predictions, and correct labels
                print(f"Incorrect indices: {incorrect_indices.tolist()}")
                print(f"Wrong predictions: {wrong_predictions.tolist()}")
                print(f"Correct labels: {correct_labels.tolist()}")

        # val_loss += criterion(val_outputs, val_labels).item()
        # _, predicted = torch.max(val_outputs, dim=1)
        # correct += (predicted == val_labels).sum().item()
        # total += val_labels.size(0)

        avg_train_loss = running_loss / len(data)
        avg_val_loss = val_loss / len(val_data)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")




# Plotting learning curves
total = (num_epochs) * 13
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# # Plotting average gradient norms
# plt.subplot(1, 2, 2)
# plt.plot(num_epochs, gradients, label='Gradient Norms')
# plt.xlabel('Epochs')
# plt.ylabel('Gradient Norm')
# plt.title('Average Gradient Norms')
# plt.legend()
# plt.tight_layout()
# plt.show()

plt.plot(gradients)
plt.xlabel('Iterations')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Magnitude over Time')
plt.show()

# fig, axs = plt.subplots(len(activation_histograms), num_epochs, figsize=(15, 10), sharex='col', sharey='row')
    
# for i, (name, activations) in enumerate(activation_histograms.items()):
#     for epoch in range(num_epochs):
#         if epoch < len(activations):
#             ax = axs[i, epoch]
#             ax.hist(activations[epoch].flatten(), bins=50, alpha=0.75)
#             ax.set_title(f'{name} Epoch {epoch+1}')
#             ax.set_xlabel('Activation Value')
#             ax.set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# Determine y-axis limits for each layer
def get_layer_y_limits(activation_histograms):
    layer_y_limits = {}
    for layer_name, histograms in activation_histograms.items():
        y_min, y_max = float('inf'), float('-inf')
        for epoch_hist in histograms:
            hist, _ = np.histogram(epoch_hist.flatten(), bins=30)
            y_min = min(y_min, hist.min())
            y_max = max(y_max, hist.max())
        layer_y_limits[layer_name] = (y_min, y_max)
    return layer_y_limits

# Activation histogram visualization
for layer_name, histograms in activation_histograms.items():
    # Get y-axis limits for each layer
    layer_y_limits = get_layer_y_limits(activation_histograms)

    plt.figure(figsize=(20, 3))
    # Set the main title for the figure
    plt.suptitle(f'{layer_name} Activation Histograms', fontsize=16)
    # Plot the histogram for each epoch
    for i, epoch_hist in enumerate(histograms):
        plt.subplot(1, len(histograms), i + 1)
        plt.hist(epoch_hist.flatten(), bins=30)
        # plt.title(f'{layer_name} Activations - Epoch {i+1}')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.ylim(layer_y_limits[layer_name])

    # plt.tight_layout()
    plt.show()

print("Training complete.")