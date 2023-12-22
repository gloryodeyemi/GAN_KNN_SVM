import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_cifar_data(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def load_cifar10_test_batch(file):
    test_data_dict = load_cifar_data(file)
    images = test_data_dict[b'data']
    labels = test_data_dict[b'labels']
    return images, labels


def merge_batches(file_prefix, num_batches):
    all_images = []
    all_labels = []
    for i in range(1, num_batches + 1):
        file_path = f'{file_prefix}_{i}'
        data_dict = load_cifar_data(file_path)

        images = data_dict[b'data']
        labels = data_dict[b'labels']

        all_images.append(images)
        all_labels += labels

    # merge the batches
    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.array(all_labels)

    return merged_images, merged_labels


def load_cifar100(train_file, test_file, meta_file):
    train_data = load_cifar_data(train_file)
    test_data = load_cifar_data(test_file)

    label_names = load_cifar_data(meta_file)[b'fine_label_names']
    label_names = [label.decode('utf-8') for label in label_names]

    train_images = train_data[b'data']
    train_fine_labels = train_data[b'fine_labels']
    test_images = test_data[b'data']
    test_fine_labels = test_data[b'fine_labels']

    return train_images, train_fine_labels, test_images, test_fine_labels, label_names


def reshape_cifar_images(images, subset_size=None):
    # flatten the images
    if subset_size is None:
        images_flat = images.reshape(images.shape[0], -1)
    else:
        images_flat = images[:subset_size].reshape(subset_size, -1)
    return images_flat


def display_sample_images(images, labels, label_names, save_alias, num_samples=10,
                          save_directory='images/data_samples'):
    num_cols = 5
    num_rows = (num_samples + num_cols - 1) // num_cols  # number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            image = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            label = label_names[labels[i]]

            ax.imshow(image)
            ax.set_title(label)
            ax.axis('off')
        else:
            ax.axis('off')  # hide extra subplots

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, f'sample_images_{num_samples}_{save_alias}.png')
    plt.savefig(save_path)
    # plt.show()
