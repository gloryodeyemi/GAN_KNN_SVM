import time

from utils.data import load_and_return_cifar10, load_and_return_cifar100
from utils.knn import KNNClassifier
from utils.preprocess import *

CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_SUBSET = 10000


def display_predictions(predictions, test_images, cifar_type, subset_size=5,
                        save_directory='images/knn_result_samples'):
    num_cols = 5
    num_rows = (subset_size + num_cols - 1) // num_cols  # number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < subset_size:
            image = test_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            ax.imshow(image)
            if cifar_type == 'CIFAR-10':
                ax.set_title(f"Predicted: {CIFAR_10_LABELS[predictions[i]]}\n"
                             f"True: {CIFAR_10_LABELS[test_labels_10[i]]}")
            else:
                ax.set_title(f"Predicted: {label_names_100[predictions[i]]}\n"
                             f"True: {label_names_100[test_fine_labels_100[i]]}")
            ax.axis('off')
        else:
            ax.axis('off')  # hide extra subplots

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, f'sample_result_{cifar_type}.png')

    plt.savefig(save_path)


def get_best_param(X_train, y_train, data_type):
    print(f"Finding best parameters for {data_type.upper()} dataset...")
    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # number of neighbors to consider
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    init_time = time.time()
    best_params = knn.grid_search(X_train, y_train, param_grid)
    cur_time = time.time()
    time_taken = cur_time - init_time
    print(f"Time taken to find best parameters: {time_taken:.4f}s")
    print(f"Best Parameters for {data_type.upper()} dataset:", best_params)
    print()


def train_without_cross_val(X_train, y_train, X_test, y_test, data_type, cifar_type='CIFAR-10'):
    print(f"{data_type.upper()} dataset:")
    print(f"Training without cross validation\n----------------------------------")
    predictions = knn.train_and_evaluate(X_train, y_train, X_test, y_test, data_type)
    if data_type == 'CIFAR-10' or data_type == 'CIFAR-100':
        display_predictions(predictions, X_test, cifar_type=cifar_type)
    return predictions


def train_with_cross_val(X_train, y_train, data_type):
    print(f"Training with cross validation\n-------------------------------")
    knn.train_and_validate(X_train, y_train, data_type)
    print()


# load and process the Iris dataset
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris()

# load and process the CIFAR-10 train and test batch
train_images_10, train_labels_10, test_images_10, test_labels_10 = load_and_return_cifar10()

# load and process the CIFAR-100 data
(train_images_100, train_fine_labels_100, test_images_100, test_fine_labels_100,
 label_names_100) = load_and_return_cifar100()

# flatten the images
train_images_10_flat = reshape_cifar_images(train_images_10, CLASS_SUBSET)
test_images_10_flat = reshape_cifar_images(test_images_10)

train_images_100_flat = reshape_cifar_images(train_images_100, CLASS_SUBSET)
test_images_100_flat = reshape_cifar_images(test_images_100)

# perform classification for KNN
print("**************\nKNN Classifier\n**************")
knn = KNNClassifier()

# train and evaluate SVM on Iris dataset
datatype_ = 'iris'
get_best_param(X_train_iris, y_train_iris, datatype_)
predictions_iris = train_without_cross_val(X_train_iris, y_train_iris, X_test_iris, y_test_iris, datatype_)
train_with_cross_val(X_train_iris, y_train_iris, datatype_)

# CIFAR-10 dataset
datatype_ = 'CIFAR-10'
get_best_param(train_images_10_flat, train_labels_10[:CLASS_SUBSET], datatype_)
predictions_10 = train_without_cross_val(train_images_10_flat, train_labels_10[:CLASS_SUBSET], test_images_10_flat,
                                         test_labels_10, datatype_, cifar_type=datatype_)
train_with_cross_val(train_images_10_flat, train_labels_10[:CLASS_SUBSET], datatype_)

# CIFAR-100 dataset
datatype_ = 'CIFAR-100'
get_best_param(train_images_100_flat, train_fine_labels_100[:CLASS_SUBSET], datatype_)
predictions_100 = train_without_cross_val(train_images_100_flat, train_fine_labels_100[:CLASS_SUBSET],
                                          test_images_100_flat, test_fine_labels_100, datatype_,
                                          cifar_type=datatype_)
train_with_cross_val(train_images_100_flat, train_fine_labels_100[:CLASS_SUBSET], datatype_)
