import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Assuming you have these from previous steps
# train_loader, val_loader = load_cifar10(batch_size=128)
# model = ShuffleNet(cfg, input_channel=3, n_classes=10).to(device)

def train_and_validate(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(loss=running_loss/total, accuracy=100.*correct/total)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                for inputs, targets in vepoch:
                    vepoch.set_description(f"Validation {epoch+1}")

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    vepoch.set_postfix(val_loss=val_loss/total, val_accuracy=100.*correct/total)

    print('Finished Training')


