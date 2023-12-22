from PIL import Image
from utils.data import load_and_return_cifar10
from utils.knn import KNNClassifier
from utils.preprocess import reshape_cifar_images
from utils.svm import SVMClassifier
import matplotlib.pyplot as plt
import numpy as np
import os

CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# function to load images and convert them to feature vectors
def load_images_and_labels(data_path):
    image_paths = sorted([os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.png')])

    images = []
    labels = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img = img.resize((32, 32))
        # convert image to RGB explicitly
        img = img.convert('RGB')
        img_array = np.array(img)
        img_vector = img_array.flatten()  # flatten the image into a feature vector
        images.append(img_vector)

        label = int(image_path.split('_')[-1].split('.')[0])  # extract label from image filename
        labels.append(label)

    return np.array(images), np.array(labels)


def display_predictions(predictions, test_images, cifar_type, classifier, subset_size=5):
    num_cols = 5
    num_rows = (subset_size + num_cols - 1) // num_cols  # number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < subset_size:
            image = test_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            ax.imshow(image)
            ax.set_title(f"Predicted: {CIFAR_10_LABELS[predictions[i]]}\n"
                             f"True: {CIFAR_10_LABELS[test_labels_10[i]]}")
            ax.axis('off')
        else:
            ax.axis('off')  # hide extra subplots

    if classifier == 'svm':
        save_directory = 'images/svm_gan_result_samples'
    else:
        save_directory = 'images/knn_gan_result_samples'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, f'sample_result_{cifar_type}.png')

    plt.savefig(save_path)
    # plt.show()


# load and preprocess CIFAR-10 data
cifar10_data_path = 'data/generated_images'
train_images, train_labels = load_images_and_labels(cifar10_data_path)
_, _, test_images_10, test_labels_10 = load_and_return_cifar10()
test_images_10_flat = reshape_cifar_images(test_images_10)

# perform classification - without cross-validation
print("**************\nSVM Classifier\n**************")
svm = SVMClassifier()
predictions_10 = svm.train_and_evaluate(train_images, train_labels, test_images_10_flat,
                                        test_labels_10, 'CIFAR-10', train_type='gan')
display_predictions(predictions_10, test_images_10, cifar_type='CIFAR-10', classifier='svm')

print("**************\nKNN Classifier\n**************")
knn = KNNClassifier()
predictions_10 = knn.train_and_evaluate(train_images, train_labels, test_images_10_flat,
                                        test_labels_10, 'CIFAR-10', train_type='gan')
display_predictions(predictions_10, test_images_10, cifar_type='CIFAR-10', classifier='knn')
