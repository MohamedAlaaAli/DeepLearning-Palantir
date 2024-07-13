from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size=128, val_split=0.1):
    """
    Load the CIFAR-10 dataset and create dataloaders for training and validation.

    Args:
        batch_size (int): Batch size for dataloaders.
        val_split (float): Proportion of the training data to be used for validation.

    Returns:
        tuple: Training and validation dataloaders.
    """
    # Define CIFAR-10 transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='data', train=True, download=False, transform=transform_train)

    # Calculate the number of validation samples
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Apply validation transformations to the validation set
    val_dataset.dataset.transform = transform_val

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


