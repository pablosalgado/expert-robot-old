# -----------------------------------------------------------------------------
# Calculate the confusion matrix.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#

import glob
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator
from sklearn.metrics import classification_report

from common import constants

CODE = 'juhm'
TIME_STEPS = 6
CLASSES = constants.LABELS[38:47]

tf.config.experimental_run_functions_eagerly(True)
model = tf.keras.models.load_model(f'{pathlib.PurePath(__file__).parent.as_posix()}/../models/trial_01/{CODE}/16/6/model')

# Deletes all extracted directories from the dataset and extract again.
shutil.rmtree(constants.MPI_WONE_AUGMENTED_DATASET_PATH, ignore_errors=True)
tf.keras.utils.get_file(
    fname=f'{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
    origin=f'https://s3.us-east-2.amazonaws.com/datasets.pablosalgado.co/lg_mpi_db/{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
    extract=True
)

# Just leaves the test data by deleting all actors/actress used in training and validation.
files = glob.glob(f'{constants.MPI_WONE_AUGMENTED_DATASET_PATH}/**/*.avi', recursive=True)
for file in files:
    if not pathlib.PurePath(file).parts[-1].startswith(CODE):
        os.remove(file)

data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

x = SlidingFrameGenerator(
    classes=CLASSES,
    glob_pattern=constants.MPI_WONE_AUGMENTED_DATASET_PATH + '/{classname}/*.avi',
    nb_frames=TIME_STEPS,
    split_val=None,
    shuffle=False,
    batch_size=1,
    target_shape=(224, 224),
    nb_channel=3,
    transformation=data_aug,
    use_frame_cache=False
)

# Extracts the actual label for each generated sequence and gets the index of each label in the CLASSES list
labels = [CLASSES.index(c) for c in [pathlib.PurePath(vi['name']).parts[-2] for vi in x.vid_info]]

# Make predictions
p = model.predict(x, verbose=1)

# Get the index of each prediction in the CLASSES list
predictions = [prediction.argmax() for prediction in p]

# Build and save the confusion matrix.
c = tf.math.confusion_matrix(labels, predictions=predictions)
np.savetxt(f'trial_01_{CODE}.csv', c.numpy(), delimiter=',')

print('\nClassification Report\n')
print(classification_report(labels, predictions, target_names=CLASSES))
