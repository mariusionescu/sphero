import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from skimage.transform import resize
import settings
from hebb import HebbianNetwork

log = settings.get_logger()

PLOT_WEIGHTS = True


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def plot(data, test, predicted, fig_size=(3, 3), save=False):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, ax_arr = plt.subplots(len(data), 3, figsize=fig_size)
    for i in range(len(data)):
        if i == 0:
            ax_arr[i, 0].set_title('Train data')
            ax_arr[i, 1].set_title("Input data")
            ax_arr[i, 2].set_title('Output data')

        ax_arr[i, 0].imshow(data[i])
        ax_arr[i, 0].axis('off')
        ax_arr[i, 1].imshow(test[i])
        ax_arr[i, 1].axis('off')
        ax_arr[i, 2].imshow(predicted[i])
        ax_arr[i, 2].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig("predictions.png")
    plt.show()


def img_processing(img):
    img = resize(img, (20, 20))
    w, h = img.shape

    # Reshape
    flatten = np.reshape(img, (w * h))
    return flatten


def array_processing(array):
    w, h = array.shape
    flatten = np.reshape(array, (w * h))
    return flatten


def create_array_data():
    train_data = [
        np.array([[0.2, 0.3], [0.1, 0.0]]),
        np.array([[0.1, 0.5], [0.2, 0.3]]),
        np.array([[0.7, 0.3], [0.1, 0.8]])
    ]

    train_data = [array_processing(d) for d in train_data]
    test_data = [
        np.array([[0.2, 0.3], [0.1, 0.0]]),
        np.array([[0.8, 0.1], [0.3, 0.7]]),
        np.array([[0.4, 0.0], [0.4, 0.3]])
    ]

    test_data = [array_processing(d) for d in test_data]

    return train_data, test_data


def create_img_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    train_data = []
    for i in range(3):
        xi = x_train[y_train == i]
        train_data.append(xi[0])
        train_data.append(xi[1])
        train_data.append(xi[2])
    train_data = [img_processing(d) for d in train_data]
    test_data = []
    for i in range(3):
        xi = x_train[y_train == i]
        test_data.append(xi[3])
        test_data.append(xi[4])
        test_data.append(xi[5])
    test_data = [img_processing(d) for d in test_data]
    return train_data, test_data


def main():

    train_data, test_data = create_img_data()

    model = HebbianNetwork(layers=5, neurons=train_data[0].shape[0])

    log.info('training')
    for data in train_data:
        for i in range(50):
            model.train(data)

    predicted_data = []

    log.info('prediction')
    for data in test_data:
        predicted = model.predict(data)
        predicted_data.append(predicted)

    log.info('plot.prediction')
    plot(train_data, test_data, predicted_data, fig_size=(50, 50))
    if PLOT_WEIGHTS:
        log.info('plot.weights')
        for layer in range(model.layers):
            model.plot_weights(layer=layer)


if __name__ == '__main__':
    main()
