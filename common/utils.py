import glob
import os
import pathlib
import random as rnd
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import constants


def save_sample(filename, g, index=0, randomize=False, row_width=22, row_height=5):
    """ Displays a batch using matplotlib.

    params:

    - g: keras video generator
    - index: integer index of batch to see (overriden if random is True)
    - randomize: boolean, if True, take a random batch from the generator
    - row_width: integer to give the figure height
    - row_height: integer that represents one line of image, it is multiplied by \
    the number of sample in batch.
    """
    total = len(g)
    if randomize:
        sample = rnd.randint(0, total)
    else:
        sample = index

    assert index < len(g)
    sample = g[sample]
    sequences = sample[0]
    labels = sample[1]

    rows = len(sequences)
    index = 1
    plt.figure(figsize=(row_width, row_height * rows), num=1, clear=True)
    for batchid, sequence in enumerate(sequences):
        classid = np.argmax(labels[batchid])
        classname = g.classes[classid]
        cols = len(sequence)
        for image in sequence:
            plt.subplot(rows, cols, index)
            plt.title(classname)
            plt.imshow(image)
            plt.axis('off')
            index += 1
    plt.savefig(filename)


def plot_acc(history, title="Model Accuracy"):
    """Imprime una gráfica mostrando la accuracy por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['accuracy'])

    if history.history.get('val_accuracy'):
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Train', 'Val'], loc='upper left')
    else:
        plt.legend(['Train'], loc='upper left')

    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.grid(True)


def plot_loss(history, title="Model Loss"):
    """Imprime una gráfica mostrando la pérdida por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['loss'])

    if history.history.get('val_accuracy'):
        plt.plot(history.history['val_loss'])
        plt.legend(['Train', 'Val'], loc='upper right')
    else:
        plt.legend(['Train'], loc='upper right')

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)


def plot_acc_loss(history, path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_acc(history)
    plt.subplot(1, 2, 2)
    plot_loss(history)
    plt.savefig(path)


def prepare_test_dataset(dataset, subdir, code) -> None:
    """
    Downloads the specified dataset to the the specified Keras cache subdirectory and deletes the videos for the actress

    :param dataset: Dataset to download
    :param subdir: Subdirectory to extract the dataset
    :param code: Code of the actor
    """
    # Remove all of the extracted directories from the dataset and extract them again.
    shutil.rmtree(f'{constants.KERAS_PATH}/{subdir}/{dataset}', ignore_errors=True)
    tf.keras.utils.get_file(
        fname=f'{dataset}.tar',
        origin=f'https://s3.us-east-2.amazonaws.com/datasets.pablosalgado.co/lg_mpi_db/{dataset}.tar',
        cache_subdir=subdir,
        extract=True
    )

    # Delete the next actress to leave her out of the training.
    files = glob.glob(f'{constants.KERAS_PATH}/{subdir}/{dataset}/**/*.avi',
                      recursive=True)
    for file in files:
        if not pathlib.PurePath(file).parts[-1].startswith(code):
            os.remove(file)
