# This file is obtained from
# https://github.com/yunjey/pytorch-tutorial

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import wandb
# Run wandb login --relogin and enter your API keys in the settings
# You only need to run once
# wandb.login()

import argparse

def main():

    config = wandb.config

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False)


    # Fully connected neural network with one hidden layer
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out


    model = NeuralNet(config['input_size'], config['hidden_size'], config['num_classes']).to(device)

    wandb.watch(model, log_graph=True)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    total_step = len(train_loader)
    for epoch in range(config['num_epochs']):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config['num_epochs'], i + 1, total_step, loss.item()))

            avg_loss += loss

        # Log the train loss
        avg_loss /= len(train_loader)
        wandb.log({"Train/Loss": avg_loss}, step=epoch)

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

        # Log the test accuracy
        wandb.log({"Test/Accuracy": 100 * correct / total})

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing argument")
    parser.add_argument("--input_size", default=784, help="Input dimension of MLP")
    parser.add_argument("--hidden_size", default=500, help="The dimension of hidden layer")
    parser.add_argument("--num_classes", default=10, help="The number of classes")
    parser.add_argument("--num_epochs", default=5, help="Number of epochs")
    parser.add_argument("--batch_size", default=100, help="The dimension of batch size")
    parser.add_argument("--learning_rate", default=0.001, help="The learning rate")

    args = parser.parse_args()

    #Initialize given the parameter
    wandb.init(project="CMPUT469-demo", config=args)

    main()