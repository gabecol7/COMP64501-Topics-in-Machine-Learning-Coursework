"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.

This training script uses imports relative to the base directory (assignment/).
To run this training script with uv, ensure you're in the root directory (assignment/)
and execute: uv run -m submission.fashion_training
"""
import os
import numpy as np
import torch, torchvision

from submission import engine
from submission.fashion_model import Net


def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=4,
                        learning_rate=0.001,
                        USE_GPU=False,):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or return values, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    """
    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create train-val split
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    # Initialize model, loss function, and optimizer
    model = Net()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.py! 
    # To use them for training or inference, we need to transform them to tensors:
    fashion_mnist.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    return fashion_mnist


def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    # TODO: create data splits

    # TODO: implement hyperparameter search

    # Train model 
    # TODO: this may be done within a loop for hyperparameter search / cross-validation
    model_weights = train_fashion_model(fashion_mnist, n_epochs=1)

    # Save model weights
    # However you tune and evaluate your model, make sure to save the final weights 
    # to submission/model_weights.pth before submission!
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()