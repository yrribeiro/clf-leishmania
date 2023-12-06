import config
import random
from Net import Net
from joblib import load
from train import train_validate
from evaluate import grid_search, naive_eval, plot_roc_curve, get_avg_metrics, get_predictions, plot_images

from sklearn.metrics import matthews_corrcoef


import torch
import numpy as np
from torchvision import datasets, transforms


np.random.seed = config.SEED
random.seed = config.SEED

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

def read_data(data_path, BATCH_SIZE, TRAIN_SIZE, VALID_SIZE, rgb=False):
    """
    Loads and splits a dataset from a specified path into training, validation, and test sets.

    Parameters:
    - data_path (str): The path to the dataset.
    - BATCH_SIZE (int): The size of the batch for data loading.
    - TRAIN_SIZE (float): Proportion of the dataset to be used for training.
    - VALID_SIZE (float): Proportion of the dataset to be used for validation.
    - rgb (bool): Flag to determine whether to load images in RGB or grayscale.

    Returns:
    Tuple[DataLoader, DataLoader, DataLoader]: Three DataLoaders for the training, validation, and test sets.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if not rgb else transforms.Resize((96, 96)),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(TRAIN_SIZE * len(dataset))
    valid_size = int(VALID_SIZE * len(dataset))

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_counts = sum(label == 0 for _, label in train_dataset)
    valid_counts = sum(label == 0 for _, label in valid_dataset)
    test_counts = sum(label == 0 for _, label in test_dataset)

    print(f'label format = {dataset.class_to_idx}')
    print(f'Dataset split: train[{len(train_dataset)}], validation[{len(valid_dataset)}], test[{len(test_dataset)}]')
    print(f'Leishmania in training set = {train_counts}')
    print(f'Leishmania in validation set = {valid_counts}')
    print(f'Leishmania in testing set = {test_counts}')

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = read_data(
    config.DPLUS_DATA_DIR,
    config.BATCH_SIZE,
    config.TRAIN_SIZE,
    config.VALID_SIZE,
    rgb=True)

for err in config.LOSSES:
    print('\n-------------------------------------------------------------------------------')
    print(f'----- LOSS FUNCTION FOR THIS ITERATION [{err.upper()}] -------------------------------')

    train_embeddings, y_true_train = train_validate(config.DATA_ID, err, train_loader, val_loader)
    grid_search(train_embeddings, y_true_train, config.DATA_ID, err)
    naive_eval(config.DATA_ID, err, test_loader)

    loaded_model = Net(config.EMBEDDING_SIZE, config.NET_IMG_SIZE).to(config.device)
    loaded_model.load_state_dict(torch.load(config.MODEL_OUT_DIR.joinpath(f'model_{config.DATA_ID}_{err}_{config.EXP_NAME}_{config.TIMESTAMP}.pth')))
    loaded_model.eval()

    loaded_clf = load(config.MODEL_OUT_DIR.joinpath(f'clf_{config.DATA_ID}_{err}_{config.EXP_NAME}.joblib'))
    loaded_pca = load(config.MODEL_OUT_DIR.joinpath(f'pca_{config.DATA_ID}_{err}_{config.EXP_NAME}.joblib'))
    loaded_scaler = load(config.MODEL_OUT_DIR.joinpath(f'scaler_{config.DATA_ID}_{err}_{config.EXP_NAME}.joblib'))

    # roc/auc curve
    plot_roc_curve(err, test_loader, loaded_model, loaded_pca, loaded_scaler)

    # stratified cross validation on test set
    get_avg_metrics(test_loader, loaded_model, loaded_scaler, loaded_clf, loaded_pca)

    # mcc
    y_true, y_pred = get_predictions(test_loader, loaded_model, loaded_pca, loaded_scaler, loaded_clf)
    print(f'MCC [{err}] = {matthews_corrcoef(y_true, y_pred)}')

    # image labelling
    plot_images(err, loaded_model, loaded_scaler, loaded_pca, loaded_clf, test_loader)