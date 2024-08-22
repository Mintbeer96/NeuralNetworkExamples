import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
torch.manual_seed(42)

# Create a simple 2D dataset with two classes
# Class 0: points around (1, 1)
# Class 1: points around (3, 3)
data = torch.tensor([[1.0, 1.0], [1.5, 1.5], [2.9, 1.0], [3.1, 1.0],[2.8, 2.8], [3.5, 3.5], [4.0, 3.0]])
labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])

test_data = torch.tensor([[-1.0, 0.0], [-1.0, -2.0], [2.6, 1.0], [2.8, 0.9],[2.9, 2.9],  [3.5, 3.5], [3.7, 2.5], [4.0, 3.0], [10, 10], [-10, -10]])
test_labels = torch.tensor([0, 0, 0, 0,1, 1, 1, 1, 1, 0])


# Visualize the dataset
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)  # Input layer to hidden layer
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 2)  # Hidden layer to output layer
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x

# Instantiate the model
model = SimpleNN()

# Using SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.019)
# Number of epochs
epochs = 5000

# Using Cross-Entropy Loss for classification

criterion = nn.CrossEntropyLoss()
# Training and validation loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()
    train_loss = 0.0
    # Forward pass

    # =========================
    # shuffle data
    # =========================

    # Generate a random permutation of indices
    indices = torch.randperm(len(data))

    # Shuffle data and labels using the generated indices
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    # =============

    outputs = model(data)
    
    # Compute the loss
    loss = criterion(outputs, labels)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    train_loss += loss.item() * data.size(0)
    train_loss /= data.size(0)
    train_losses.append(train_loss)

# Plotting learning curves
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Using Mean Squared Error (MSE) Loss instead of Cross-Entropy

# criterion = nn.MSELoss()

# # Training the model again with the new loss function
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     outputs = model(data.float())
#     train_loss = 0.0
#     one_hot_labels = nn.functional.one_hot(labels, num_classes=2).float()  # Convert labels to one-hot
#     loss = criterion(outputs, one_hot_labels)
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

#     train_loss += loss.item() * data.size(0)
#     train_loss /= data.size(0)
#     train_losses.append(train_loss)

# # Plotting learning curves
# plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# Evaluate the model on the training data
with torch.no_grad():
    predicted = model(test_data)
    _, predicted_labels = torch.max(predicted, 1)
    
    print(f'Predicted Labels: {predicted_labels}')
    print(f'True Labels: {test_labels}')

    # Visualize the decision boundary
    plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_labels, cmap='coolwarm', marker='x')
    node = torch.tensor([[1,1], [3,3]])
    plt.scatter(node[:, 0], node[:, 1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

import matplotlib.pyplot as plt

def draw_neural_network(ax, layer_sizes):
    """
    Draw a simple neural network diagram.
    
    Parameters:
    - ax: The matplotlib axis where the network will be drawn.
    - layer_sizes: A list of integers where each integer represents the number of neurons in that layer.
    """
    v_spacing = 1.0 / max(layer_sizes)
    h_spacing = 1.0 / (len(layer_sizes) - 1)
    
    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing, 1 - j * v_spacing), v_spacing / 4., color='black', fill=True)
            ax.add_artist(circle)
    
    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [1 - j * v_spacing, 1 - k * v_spacing], color='black')
                ax.add_artist(line)

# Example network
layer_sizes = [2, 10, 10, 10, 10, 2]  # Example: 2 neurons in input layer, 3 in hidden layer, 2 in output layer

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')  # Turn off the axis
draw_neural_network(ax, layer_sizes)
plt.show()
