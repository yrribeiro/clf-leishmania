import config
from Net import Net
import numpy as np
import matplotlib.pyplot as plt

import torch
from pytorch_metric_learning import distances, losses, reducers

##############################################################################################
## Auxiliary functions

def train(model, data_loader, optimizer, loss_function):
    """
    Trains a neural network model using a given data loader, optimizer, and loss function.

    Parameters:
    - model (torch.nn.Module): The neural network model to train.
    - data_loader (DataLoader): DataLoader for the training data.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - loss_function (function): Loss function used for training.

    Returns:
    float: The total training loss after completing the training over the dataset.
    """
    model.train()
    train_loss = 0
    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss

def validate(model, data_loader, loss_function):
    """
    Validates a neural network model using a given data loader and loss function.

    Parameters:
    - model (torch.nn.Module): The neural network model to validate.
    - data_loader (DataLoader): DataLoader for the validation data.
    - loss_function (function): Loss function used for validation.

    Returns:
    float: The total validation loss after evaluating the dataset.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            output = model(inputs)
            loss = loss_function(output, labels)

            val_loss += loss.item()

    return val_loss

def extract_embeddings(model, data_loader):
    """
    Extracts embeddings from a dataset using a trained model.

    Parameters:
    - model (torch.nn.Module): The trained neural network model.
    - data_loader (DataLoader): DataLoader for the data.

    Returns:
    Tuple[np.array, np.array]: Arrays of embeddings and corresponding labels.
    """
    model.eval()
    embeddings = []
    labels_list = []

    # a progress bar could be handy here TODO
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            output = model(inputs)
            embeddings.append(output)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels_list).cpu().numpy()

    return embeddings, labels


def save_model(loss_select, model, data_id):
    """
    Saves the trained model state to a .pth file.

    Parameters:
    - loss_select (str): Name of the loss function used during training.
    - model (torch.nn.Module): The trained neural network model to save.
    - data_id (str): Identifier for the dataset used in training.

    Returns:
    None
    """
    model_filename = config.MODEL_OUT_DIR.joinpath(f'model_{data_id}_{loss_select}_{config.EXP_NAME}_{config.TIMESTAMP}.pth')
    torch.save(model.state_dict(), model_filename)

################################################################################################################
## Main function

def train_validate(data_id, loss_select, train_loader, valid_loader):
    """
    Trains and validates a neural network model using specified loss functions and data loaders.

    Parameters:
    - data_id (str): Identifier for the dataset.
    - loss_select (str): Loss function to be used for training.
    - train_loader (DataLoader): DataLoader for the training data.
    - valid_loader (DataLoader): DataLoader for the validation data.

    Returns:
    Tuple[np.array, np.array]: Arrays of training embeddings and labels.
    """
    loss_function = None
    if loss_select == 'Triplet': # default is euclidean but changed to cosine
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_function = losses.TripletMarginLoss(margin=0.3, distance=distance, reducer=reducer)

    if loss_select == 'NPairs': # uses dot product that becomes cosine sim, normalize_embed = true
        loss_function = losses.NPairsLoss()

    if loss_select == 'Circle':
        loss_function = losses.CircleLoss() # uses cosine sim

    if loss_select == 'MultiSimilarity': # uses cosine sim
        loss_function = losses.MultiSimilarityLoss(alpha = 2, beta = 50, base=0.5)

    net = Net(config.EMBEDDING_SIZE, config.NET_IMG_SIZE).to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.INIT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter, best_epoch = 0, 0

    for epoch in range(config.EPOCHS):
        train_loss = train(net, train_loader, optimizer, loss_function)
        val_loss = validate(net, valid_loader, loss_function)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch + 1}/{config.EPOCHS} - '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}, ')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            save_model(loss_select, net, data_id)
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5, label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(config.PLOTS_MODEL_EVAL_OUT.joinpath(f'train_metrics_{data_id}_{loss_select}.png'), bbox_inches='tight')
    plt.show()

    train_embeddings, train_labels = extract_embeddings(net, train_loader)
    val_embeddings, val_labels = extract_embeddings(net, valid_loader)

    np.save(config.MODEL_OUT_DIR.joinpath(f'train_embeddings_{data_id}_{loss_select}.npy'), train_embeddings)
    np.save(config.MODEL_OUT_DIR.joinpath(f'train_labels_{data_id}_{loss_select}.npy'), train_labels)
    np.save(config.MODEL_OUT_DIR.joinpath(f'val_embeddings_{data_id}_{loss_select}.npy'), val_embeddings)
    np.save(config.MODEL_OUT_DIR.joinpath(f'val_labels_{data_id}_{loss_select}.npy'), val_labels)

    # save_model(loss_select, net, data_id)

    return train_embeddings, train_labels