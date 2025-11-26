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

from copy import deepcopy
from submission import engine
from submission.fashion_model import Net
from matplotlib import pyplot as plt

def k_fold_split(fashion_mnist, k=5):
    """
    Partitions dataset k disjoint folds of the data to be left out of the overall training
    dataset.
    """

    #Calculate size of folds as well as create array to store each subset
    fold_size = int(len(fashion_mnist)//k)
    fold_tuples = []

    #Use index list to create disjoint partitions
    dataset_index = np.arange(len(fashion_mnist))
    np.random.shuffle(dataset_index)

    for fold in range(k):


        #Partition the dataset indices into disjoint folds, remove these fold indices from overall indices to determine train indicies
        val_index = dataset_index[fold*fold_size:(fold+1)*fold_size]
        train_index = np.delete(dataset_index, np.arange(fold*fold_size - 1, (fold+1)*fold_size - 1))

        #Create fold subsets
        val_fold = torch.utils.data.Subset(fashion_mnist, val_index)
        train_fold = torch.utils.data.Subset(fashion_mnist, train_index)

        #Append tuple of training data and validation data 
        fold_tuples.append((train_fold, val_fold))
    
    return fold_tuples


def cross_validation(fashion_mnist,
                    n_epochs=15,
                    k=5,
                    batch_size=4,
                    learning_rate=0.001,
                    model = Net(),
                    USE_GPU=True,):
    """
    Used to get validation metrics for all k-folds and return a mean result for the
    cross-validation loss
    """
    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    #Generate a list of tuples for the k-folds
    fold_tuples = k_fold_split(fashion_mnist, k)

    #Save initial model state, loss function, and optimizer
    initial_model_state = deepcopy(model.state_dict())
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Use LR scheduler based on validation loss metric, (Patience set to 2, but occurs after 3 iterations of not improving)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=2,
                                                           threshold=1e-4)
        
    #Initialise early stopping parameters
    patience = 5
    no_improvement_count = 0
    threshold = 1e-4 #Amount validation loss needs to decrease to past best
    best_val_loss = np.inf
    best_model_state = None
    
    #Initialise list for the val loss at each fold

    val_loss_list = []
    accuracy_list = []

    # Training loop
    for fold in range(len(fold_tuples)):
        
        #Load initial model state and set best_model_state to None
        model.load_state_dict(initial_model_state)
        best_model_state = None
        no_improvement_count = 0
        best_val_loss = np.inf

        #Reset learning rate parameter before training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Load Fold Data
        train_loader = torch.utils.data.DataLoader(fold_tuples[fold][0],
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_loader = torch.utils.data.DataLoader(fold_tuples[fold][1],
                                                 batch_size=batch_size,
                                                 shuffle=True)

        for epoch in range(n_epochs):
            
            #Start training
            engine.train(model, train_loader, criterion, optimizer, device)
            val_loss, _ = engine.eval(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            #Early stopping implementation
            if best_model_state == None:
                
                #Initialise best validation loss and initial model state
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict()) #Deepcopy required to prevent changing with updating model
                no_improvement_count = 0
                
            elif val_loss < best_val_loss - threshold:
                
                #Update new best validation loss and reset counter
                best_val_loss = val_loss
                best_model_state =  deepcopy(model.state_dict()) #Deepcopy required to prevent changing with updating model
                no_improvement_count = 0
                
            else:
                
                #Update count for number of iterations without improvement
                no_improvement_count += 1
                
            
            #End loop early if early stopping criterion occurs
            if no_improvement_count >= patience:
                break
            
        #Load best model state and evaluate validation loss
        model.load_state_dict(best_model_state)
        val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
        
        #Append validation loss
        print(f"Fold [{fold + 1}/{k}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)

    return np.mean(val_loss_list), np.std(val_loss_list, ddof=1), np.mean(accuracy_list), np.std(accuracy_list, ddof = 1)



def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=64,
                        learning_rate=0.001,
                        save_figure=False,
                        USE_GPU=True,):
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

    #Use LR scheduler based on validation loss metric, (Patience set to 2, but occurs after 3 iterations of not improving)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=2,
                                                           threshold=1e-4)
    
    #Initialise early stopping parameters
    patience = 5
    no_improvement_count = 0
    threshold = 1e-4 #Amount validation loss needs to decrease to past best
    best_val_loss = np.inf
    best_model_state = None
    
    
    #Keep track of train/val loss and accuracy at each epoch if ploting of figures is required
    if save_figure:
        train_loss_epoch = []
        train_accuracy_epoch = []
        val_loss_epoch = []
        val_accuracy_epoch = []

    # Training loop
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}. Learning Rate: {optimizer.param_groups[0]['lr']}")
        val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if save_figure:
            _, train_accuracy = engine.eval(model, train_loader, criterion, device)
            train_loss_epoch.append(train_loss)
            train_accuracy_epoch.append(train_accuracy)
            val_loss_epoch.append(val_loss)
            val_accuracy_epoch.append(accuracy) #Accuracy is val_accuracy


        #Step scheduler based upon the validation loss found
        scheduler.step(val_loss)
        
        #Early stopping implementation
        if best_model_state == None:
            
            #Initialise best validation loss and initial model state
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict()) #Deepcopy required to prevent changing with updating model
            no_improvement_count = 0
            
        elif val_loss < best_val_loss - threshold:
            
            #Update new best validation loss and reset counter
            best_val_loss = val_loss
            best_model_state =  deepcopy(model.state_dict()) #Deepcopy required to prevent changing with updating model
            no_improvement_count = 0
            
        else:
            
            #Update count for number of iterations without improvement
            no_improvement_count += 1
            
        
        #End loop early if early stopping criterion occurs
        if no_improvement_count >= patience:
            break
        
    #Load best model state and print final val loss and accuracy
    
    model.load_state_dict(best_model_state)
    val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
    print(f"Final Val Loss: {val_loss:.4f}, Final Accuracy: {accuracy:.4f}")
    #Plot test and validation loss/accuracy curves if plot_figures is set to True
    if save_figure:
        plt.figure(figsize=(16,5))
        plt.subplot(121)

        plt.plot(np.arange(1,len(train_loss_epoch)+1), np.array(train_loss_epoch), label='Training Loss')
        plt.plot(np.arange(1,len(val_loss_epoch)+1), np.array(val_loss_epoch), label='Validation Loss')

        plt.title('Cross-entropy Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-entropy Loss')
        plt.legend();

        plt.subplot(122)

        plt.plot(np.arange(1,len(train_accuracy_epoch)+1), np.array(train_accuracy_epoch), label='Training Accuracy')
        plt.plot(np.arange(1,len(val_accuracy_epoch)+1), np.array(val_accuracy_epoch), label='Validation Accuracy')

        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend();
        plt.tight_layout()
        plt.savefig('train_curves.png')

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions), please check that 
    these pass by running model_calls.py before submission. Transforms will be set to .eval()
    (deterministic) mode during evaluation, so avoid using stochastic transforms like RandomCrop
    or RandomHorizontalFlip unless they can be set to p=0 during eval.
    """
    if mode == 'train':
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
            torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ])
    elif mode == 'eval': # no stochastic transforms, or use p=0
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
            torchvision.transforms.RandomHorizontalFlip(p=0)
        ])
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


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
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist


def main():

    #Load data
    fashion_mnist = load_training_data()

    #Bool to determine to run cross_validation function or train model
    cross_val = False
    train_model = True
    save_figure = True #Bool to determine to plot loss/accuracy curves


    #Taken from hyperparameter optimisation
    batch_size = 32
    lr = 0.008481069504739685

    #Run cross-validation 
    if cross_val:
        val_loss_mean, val_loss_std, val_accuracy_mean, val_accuracy_std = cross_validation(fashion_mnist,
                                                                                            n_epochs=50,
                                                                                            k=5,
                                                                                            batch_size=batch_size,
                                                                                            learning_rate=lr,
                                                                                            model = Net())

        print(f'Mean Validation Loss: {val_loss_mean:.4f}, Validation Loss Sample Standard Deviation: {val_loss_std:.4f}')
        print(f'Mean Validation Accuracy: {val_accuracy_mean:.4f}, Validation Accuracy Sample Standard Deviation: {val_accuracy_std:.4f}')

    
    # Train model 
    if train_model:
        model_weights = train_fashion_model(fashion_mnist, n_epochs=50, batch_size = batch_size, learning_rate = lr, save_figure=save_figure)
        
        # Save model weights
        # However you tune and evaluate your model, make sure to save the final weights 
        # to submission/model_weights.pth before submission!
        model_save_path = os.path.join('submission', 'model_weights.pth')
        torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()