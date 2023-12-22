from utils.preprocess import *

CIFAR_10_FILE_prefix = 'data/cifar-10-batches-py/data_batch'
CIFAR_10_NUM_BATCHES = 5
CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR_10_TEST_FILE = 'data/cifar-10-batches-py/test_batch'

CIFAR_100_TRAIN_FILE = 'data/cifar-100-python/train'
CIFAR_100_TEST_FILE = 'data/cifar-100-python/test'
CIFAR_100_META_FILE = 'data/cifar-100-python/meta'


def load_and_return_cifar10():
    # load and process the CIFAR-10 train and test batch
    train_images_10, train_labels_10 = merge_batches(CIFAR_10_FILE_prefix, CIFAR_10_NUM_BATCHES)
    display_sample_images(train_images_10, train_labels_10, CIFAR_10_LABELS, save_alias='train_10')
    test_images_10, test_labels_10 = load_cifar10_test_batch(CIFAR_10_TEST_FILE)
    display_sample_images(test_images_10, test_labels_10, CIFAR_10_LABELS, save_alias='test_10')
    return train_images_10, train_labels_10, test_images_10, test_labels_10


def load_and_return_cifar100():
    # load and process the CIFAR-100 data
    train_images_100, train_fine_labels_100, test_images_100, test_fine_labels_100, label_names_100 = load_cifar100(
        CIFAR_100_TRAIN_FILE, CIFAR_100_TEST_FILE, CIFAR_100_META_FILE)
    display_sample_images(train_images_100, train_fine_labels_100, label_names_100, save_alias='train_100')
    display_sample_images(test_images_100, test_fine_labels_100, label_names_100, save_alias='test_100')
    return train_images_100, train_fine_labels_100, test_images_100, test_fine_labels_100, label_names_100
