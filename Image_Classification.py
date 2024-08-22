import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import numpy as np
# Your main function
def main():
    # Get the test dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Define the basic building block of ResNet, called 'BasicBlock'
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(BasicBlock, self).__init__()

            # First convolutional layer
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            # Batch normalization after the first convolution
            self.bn1 = nn.BatchNorm2d(out_channels)
            # ReLU activation function
            self.relu = nn.ReLU(inplace=True)

            # Second convolutional layer
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # Batch normalization after the second convolution
            self.bn2 = nn.BatchNorm2d(out_channels)

            # Downsampling layer, if needed, to match the dimensions
            self.downsample = downsample

        def forward(self, x):
            # Save the input (identity) for the residual connection
            identity = x

            # Apply the first convolution, followed by batch normalization and ReLU
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            # Apply the second convolution, followed by batch normalization
            out = self.conv2(out)
            out = self.bn2(out)

            # If there's a downsample layer, apply it to the identity to match dimensions
            if self.downsample is not None:
                identity = self.downsample(x)

            # Add the original input (identity) to the output (residual connection)
            out += identity
            # Apply ReLU to the final output
            out = self.relu(out)

            return out

    # Define the entire ResNet-like model, called 'LittleResNet'
    class LittleResNet(nn.Module):
        def __init__(self, num_classes=2):
            super(LittleResNet, self).__init__()

            # Initial convolutional layer that processes the input image
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Create four layers with different channel sizes and strides
            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)

            # Adaptive average pooling to reduce the spatial dimensions to 1x1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # Fully connected layer to produce the final classification output
            self.fc = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(0.5)

        # A helper function to create a layer with multiple BasicBlocks
        def _make_layer(self, out_channels, blocks, stride=1):
            # Downsample if the stride is not 1 or if the number of input and output channels don't match
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

            # Create a list of layers starting with the first block that may include downsampling
            layers = []
            layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels

            # Add additional blocks without downsampling
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_channels, out_channels))

            # Return the layer as a sequential container
            return nn.Sequential(*layers)

        def forward(self, x):
            # Pass the input through the initial convolutional layer and max pooling
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Pass through each of the four layers
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            # Perform adaptive average pooling to reduce the output to 1x1
            x = self.avgpool(x)
            # Flatten the output to prepare it for the fully connected layer
            x = torch.flatten(x, 1)
            x = self.dropout(x) # Applying dropout before the final output layer
            # Pass through the fully connected layer to get the final classification
            x = self.fc(x)

            return x


    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.5)
            
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            
            # Calculate the output size after all conv and pooling layers
            self.fc1 = nn.Linear(16 * 16 * 256, 512)  # Adjusting based on output size after conv layers
            self.bn5 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 2)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            
            x = x.view(x.size(0), -1)  # Flattening
            x = self.fc1(x)
            x = self.bn5(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x

    # Define the CNN model
    class CatsVsDogsCNN(nn.Module):
        def __init__(self):
            super(CatsVsDogsCNN, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            # Batch normalization
            self.batch_norm1 = nn.BatchNorm2d(32)
            self.batch_norm2 = nn.BatchNorm2d(64)
            self.batch_norm3 = nn.BatchNorm2d(128)
            
            # Pooling layers
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
            # Fully connected layers
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 2)
            # self.softmax = nn.Softmax()

            # Dropout for regularization
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            # Convolutional layers with ReLU, Batch Norm, and Pooling
            x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
            x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
            x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
            
            # Flatten the tensor
            x = x.view(-1, 128 * 28 * 28)
            
            # Fully connected layers with ReLU and Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            # Output layer
            x = self.fc4(x)
            # X = self.softmax(X)

            return x
    
    class ImprovedConvNet(nn.Module):
        def __init__(self):
            super(ImprovedConvNet, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channels, 16 output channels
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 input channels, 32 output channels
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels, 64 output channels
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Adding a fourth conv layer with 128 channels
            
            # Batch Normalization
            self.batch_norm1 = nn.BatchNorm2d(16)
            self.batch_norm2 = nn.BatchNorm2d(32)
            self.batch_norm3 = nn.BatchNorm2d(64)
            self.batch_norm4 = nn.BatchNorm2d(128)
            
            # Dropout layer
            self.dropout = nn.Dropout(0.5)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust input size based on the final conv layer output size
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 2)  # Output layer for binary classification

             # Initialize weights
            self._initialize_weights()

        def forward(self, x):
            x = F.relu(self.batch_norm1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.batch_norm3(self.conv3(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.batch_norm4(self.conv4(x)))  # Apply the fourth convolution layer
            x = F.max_pool2d(x, 2)
            
            x = x.view(x.size(0), -1)  # Flatten the output
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # Apply dropout after first fully connected layer
            x = F.relu(self.fc2(x))
            x = self.dropout(x)  # Apply dropout after second fully connected layer
            x = self.fc3(x)  # Final output layer
            return x
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
    # Initialize the model, loss function, and optimizer
    # model = CatsVsDogsCNN().to(device)
    # Create an instance of the model with two output classes
    model = LittleResNet(num_classes=2).to(device)

    activations = dict()

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
        # Define a function to register hooks on all layers of the model
    def register_hooks(model):
        for name, layer in model.named_modules():
            # Register the hook only on layers with learnable parameters (e.g., Conv2d, Linear)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_forward_hook(get_activation(name))
    # # Register hooks on the layers you want to monitor
    # model.conv1.register_forward_hook(get_activation('conv1'))
    # model.conv2.register_forward_hook(get_activation('conv2'))
    # model.conv3.register_forward_hook(get_activation('conv3'))
    # # model.conv4.register_forward_hook(get_activation('conv4'))
    # model.fc1.register_forward_hook(get_activation('fc1'))
    # model.fc2.register_forward_hook(get_activation('fc2'))
    # model.fc2.register_forward_hook(get_activation('fc3'))
    
    register_hooks(model)

    criterion = nn.CrossEntropyLoss()
    initial_lr = 1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr,  weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    # warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    batch_size = 32

    # # Load the dataset
    # dataset = datasets.ImageFolder(root='catsndogs/train', transform=initial_transform)

    # # Create a DataLoader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # # Compute mean and std
    # mean, std = compute_mean_std(dataloader)
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")

    # Define the final transform with normalization
    final_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a common size (optional)
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with computed mean and std
    ])

    # Data transformation and augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Reload the dataset with the final transform
    # train_dataset = datasets.ImageFolder(root='catsndogs/train', transform=data_transforms['train'])
    # val_dataset = datasets.ImageFolder(root='catsndogs/train', transform=data_transforms['val'])
    # Define validation DataLoader
    
    def calculate_accuracy(predictions, labels):
        _, preds = torch.max(predictions, 1)  # Get the index of the max log-probability
        correct = (preds == labels).sum().item()  # Count the correct predictions
        accuracy = correct / labels.size(0)  # Calculate accuracy
        return accuracy

    # Define transforms for the data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the images to 224x224 as expected by ResNet
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),      # Converting the images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing based on ImageNet standards
    ])
    dataset = datasets.ImageFolder(root='PetImages', transform=transform)

    # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_losses = []
    val_losses = []
    gradients = []
    activation_histograms = {}
    # Training loop
    accuracy = []

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_data, batch_labels in train_loader: 
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
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

            train_loss += loss.item() * batch_data.size(0)

        if len(activation_histograms.keys())  == 0:
            # Initialize lists to store activations histograms per epoch
            activation_histograms = {name: [] for name in activations.keys()}

        # Store activation histograms for this batch
        for name, activation in activations.items():
            activation_histograms[name].append(activation.cpu().numpy())

            
        # # # Update learning rate
        # scheduler.step()

        
        train_loss /= len(train_loader.dataset)  # Normalizing by the dataset size
        train_losses.append(train_loss)

        '''
        After all batches are processed, the model has gone through the entire dataset, 
        and you can assess how well the model has generalized by calculating metrics like validation accuracy or loss.
        '''
        # After training for one epoch
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(output.data, 1)
                # total += targets.size(0)
                correct += (predicted == targets).sum().item() 

            accuracy.append(correct/len(val_loader.dataset))

            val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            # scheduler.step(metrics = val_loss)
            # Print loss values
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Plotting
    plt.figure(figsize=(12, 6))

    # Training Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - lr {initial_lr}')
    plt.legend()
    plt.grid(True)

    # Validation Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracy, label='Accuracy', color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(gradients)
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude over Time')
    plt.show()

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

if __name__ == '__main__':
    main()