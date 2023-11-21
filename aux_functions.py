import torch
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms

def generate_data(input_imgs, out_folder):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=120,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        preprocessing_function=lambda x: x/255
    )
    for i in input_imgs:
        image = np.expand_dims(plt.imread(i), 0)
        datagen.fit(image)
        # './semana18/all_patches_splitted/val/leish/'
        for x, val in zip(datagen.flow(image,
            save_to_dir=out_folder,
            save_prefix='aug_',
            save_format='png'),range(13)):
            pass

    return 0

def read_data(data_path, BATCH_SIZE, TRAIN_SIZE):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(TRAIN_SIZE * len(dataset))
    # test_size = len(dataset) - train_size
    indices = torch.randperm(len(dataset)).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Obtendo as etiquetas verdadeiras do test_dataset usando Subset
    train_true_labels = [dataset.targets[idx] for idx in train_indices]
    test_true_labels = [dataset.targets[idx] for idx in test_indices]

    leish_train = sum(label == 0 for _, label in train_dataset)
    leish_test = sum(label == 0 for _, label in test_dataset)

    print(f'label format = {dataset.class_to_idx}')
    print(f'train test split proportion = train[{len(train_dataset)}], test[{len(test_dataset)}]')
    print(f'leish in training set = {leish_train}')
    print(f'leish in testing set = {leish_test}')

    return train_dataset, test_dataset, train_loader, test_loader, train_true_labels, test_true_labels

if __name__ == '__main__':
    from pathlib import Path
    DATA_DIR = Path('./data/v1-and-fba/')
    leish_pt = DATA_DIR.joinpath('leish').glob('*.png')
    out_aug_leish_pt = DATA_DIR.joinpath('aug') # visualize image quality before merging with real images

    out_aug_leish_pt.mkdir(parents=True, exist_ok=False)
    generate_data(leish_pt, out_aug_leish_pt)