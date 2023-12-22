import os
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot

CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# create and save a plot of generated images
def save_generated_images(examples, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(examples)):
        label = labels[i]
        filename = os.path.join(output_dir, f'generated_image_{i}_label_{label}.png')
        pyplot.imsave(filename, examples[i])


# create and save a plot of generated images
def save_plot(examples, labels):
    # display a subset of the generated images with labels in a subplot
    subset_samples = 10
    num_rows, num_cols = 2, 5
    subset_images = examples[:subset_samples]
    subset_labels = labels[:subset_samples]

    fig, axes = pyplot.subplots(num_rows, num_cols, figsize=(8, 4))
    for i in range(subset_samples):
        row = i // num_cols
        col = i % num_cols

        axes[row, col].imshow(subset_images[i])
        axes[row, col].axis('off')
        axes[row, col].set_title(CIFAR_10_LABELS[subset_labels[i]], fontsize=8)

    pyplot.tight_layout()

    save_directory = 'images/gan_sample'
    filename = f'generated_sample_gan.png'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, filename)
    pyplot.savefig(save_path)
    pyplot.show()


# load model
model = load_model('models/cgan_generator.keras')
# generate images
latent_points, labels = generate_latent_points(100, 10000)
# specify labels
# labels = asarray([x for _ in range(10) for x in range(10)])
# generate images
X = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0

# save generated images for classification later
output_directory = 'data/generated_images'
save_generated_images(X, labels, output_directory)

# plot the result
save_plot(X, labels)
