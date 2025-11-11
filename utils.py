"""
Utility functions for the assignment. DO NOT MODIFY THIS FILE.
This script will be replaced during marking, so any changes you make will not be saved! 
"""
import numpy as np
import torch, torchvision
from torch import nn

from submission.fashion_training import train_fashion_model


def model_check(model, device):
    """
    Small script that checks if the model as defined can perform a single
    training step on dummy data. This script will be used during marking.
    
    This checks:
    - Forward pass works
    - Backward pass works
    - Model has trainable parameters
    
    Returns:
        bool: True if sanity check passes, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:
        model.train()
        
        # Create dummy dataset (3 samples)
        dummy_data = torch.randn(3, 1, 28, 28).to(device)
        dummy_labels = torch.tensor([0, 1, 2]).to(device)
        
        # Forward pass
        outputs = model(dummy_data)
        
        # Check output shape
        if outputs.shape != (3, 10):
            return False, f"Expected output shape (3, 10), got {outputs.shape}"
        
        # Note this does not need to be the same as loss used in training,
        # just using CrossEntropyLoss as a standard check
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            return False, f"Loss is {loss.item()}, expected finite value"
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for at least some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if not has_grad:
            return False, "No gradients computed during backward pass"
        
        return True, None
        
    except Exception as e:
        return False, f"Model check failed: {str(e)}"
    

def training_check(model_class, device):
    """
    Verify train_fashion_model() runs and returns valid state_dict.
    Uses small dummy dataset (30 samples) and 1 epoch.
    
    This checks:
    - train_fashion_model() can run without errors
    - Returns a valid state_dict (dict type)
    - Returned state_dict can be loaded into model
    - Training code matches submitted architecture
    
    Returns:
        bool: True if integrity check passes, False otherwise
        str: Error message if check fails, None otherwise
    """
    try:
        # Create small dummy data (30 samples, 28x28 grayscale, 10 classes)
        dummy_images = np.random.randint(0, 256, size=(30, 28, 28), dtype=np.uint8)
        dummy_labels = [i % 10 for i in range(30)]
        
        # Create a mock dataset that behaves like FashionMNIST
        class DummyDataset:
            def __init__(self, images, labels):
                self.data = images
                self.targets = labels
                self.transform = torchvision.transforms.ToTensor()
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                image = self.data[idx]
                label = self.targets[idx]
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        dummy_dataset = DummyDataset(dummy_images, dummy_labels)
        
        # USE_GPU=False to ensure consistent behavior
        result = train_fashion_model(dummy_dataset, n_epochs=1, USE_GPU=False)
        
        # Verify return type is dict (state_dict)
        if not isinstance(result, dict):
            return False, f"train_fashion_model() returned {type(result).__name__}, should be dict (state_dict)"
        
        # Verify state_dict is not empty
        if not result:
            return False, "train_fashion_model() returned empty state_dict"
        
        # Verify we can load the state_dict into a fresh model
        test_model = model_class()
        test_model.load_state_dict(result)
        test_model.to(device)
        
        # Verify the loaded model can do a forward pass
        test_input = torch.randn(2, 1, 28, 28).to(device)
        test_output = test_model(test_input)
        
        if test_output.shape != (2, 10):
            return False, f"Model with trained weights produced wrong output shape: {test_output.shape}, expected (2, 10)"
        
        return True, None
        
    except Exception as e:
        return False, f"Training integrity check failed: {str(e)}"
