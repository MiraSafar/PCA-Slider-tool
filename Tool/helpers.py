import numpy as np
from sklearn.decomposition import PCA
from settings import *
from tensorflow import keras


def calculate_pca(latent_space: np.ndarray):
    pca = PCA(n_components=10)
    pca_encondings = pca.fit_transform(latent_space)
    pca_encondings_transposed = pca_encondings.T
    pca_info = []

    for i, pca_item in enumerate(pca_encondings_transposed):
        pca_values = [f"PCA {i + 1}"]

        maximum = np.argmax(pca_item)
        pca_values.append(pca_encondings_transposed[i][maximum])

        minimum = np.argmin(pca_item)
        pca_values.append(pca_encondings_transposed[i][minimum])

        step = (abs(pca_encondings_transposed[i][maximum]) + abs(pca_encondings_transposed[i][minimum])) / 100
        pca_values.append(step)

        pca_info.append(pca_values)

    return pca_info, pca_encondings, pca


def pca_to_pixels(pca_value: float, pca_max: float, pca_min: float):
    """
    Converts the value of a specific pca for a particular MNIST digit to a pixel value. This pixel value corresponds to
    the number of pixels the slider handle will be offset from the beginning of the slider rectangle (i.e. left_pos)

    Keyword arguments:
    pca_value -- pca encoding value obtained from the latent space PCA calculation
    pca_min -- minimum value of that particular pca in the entire latent space dataset
    pca_max -- maximum value of that particular pca in the entire latent space dataset
    """
    slope = (pca_max - pca_min)/SLIDER_WIDTH
    intercept = pca_min

    pixel_value = (pca_value - intercept)/slope

    return pixel_value


def pixels_to_pca(pixel_value: float, pca_max: float, pca_min: float):
    """
    Converts the value of a pixel position of the slider handle back to it's corresponding pca encoding value.

    Keyword arguments:
    pixel_value -- pixel position of the slider handle
    pca_min -- minimum value of that particular pca in the entire latent space dataset
    pca_max -- maximum value of that particular pca in the entire latent space dataset
    """

    slope = (pca_max - pca_min)/SLIDER_WIDTH
    intercept = pca_min

    pca_value = (pixel_value * slope) + intercept

    return pca_value


def load_mnist_full():

    # Load MNIST again to see what pictures does this correspond to:
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Merge the training and test MNIST data into one array
    complete_mnist_dataset = np.concatenate((X_train_full, X_test), axis=0)

    return complete_mnist_dataset


