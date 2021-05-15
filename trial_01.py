# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#

import glob
import os
import shutil

import pandas
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

from common import constants, utils

# Avoids the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2 GeForce RTX 2080
# SUPER GPU cards.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parameters
TRIAL = '01'
BATCH_SIZE = [32, 64]
TIME_STEPS = [12, 24]
EPOCHS = 1000

TRL_PATH = f'models/trial_{TRIAL}'


def build_model(time_steps, nout):
    # Load MobileNet model excluding top.
    cnn_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )

    # Allows to retrain the last convolutional layer.
    for layer in cnn_model.layers[:-3]:
        layer.trainable = False

    # Build the new CNN adding a layer to flatten the convolution as required
    # for the RNN.
    cnn_model = tf.keras.models.Sequential(
        [
            cnn_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ]
    )

    # Now build the RNN model.
    rnn_model = tf.keras.models.Sequential()

    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(time_steps, 224, 224, 3)))

    # Build the classification layer.
    rnn_model.add(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.5))
    rnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))

    rnn_model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    rnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))

    rnn_model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    rnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))

    rnn_model.add(tf.keras.layers.LSTM(64))
    rnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))

    rnn_model.add(tf.keras.layers.Dense(nout, activation='softmax'))

    rnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    return rnn_model


def train():
    data = pandas.DataFrame(None, columns=['trial', 'cycle', 'code', 'batch_size', 'time_steps', 'files', 'sequences'])
    data['trial'] = TRIAL

    for code in constants.CODES:
        # Deletes all extracted directories from the dataset and extract again.
        shutil.rmtree(constants.MPI_WONE_AUGMENTED_DATASET_PATH, ignore_errors=True)
        tf.keras.utils.get_file(
            fname=f'{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
            origin=f'https://s3.us-east-2.amazonaws.com/datasets.pablosalgado.co/lg_mpi_db/{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
            extract=True
        )

        # So it can be deleted the next actor/actress to leave out him/her in training
        files = glob.glob(f'{constants.MPI_WONE_AUGMENTED_DATASET_PATH}/**/{code}*.avi', recursive=True)
        for file in files:
            os.remove(file)

        for batch_size in BATCH_SIZE:
            for time_steps in TIME_STEPS:
                path = TRL_PATH + f'/{code}/{batch_size}/{time_steps}'
                os.makedirs(path, exist_ok=True)

                # Build and compile the model.
                model = build_model(time_steps, len(constants.LABELS[38:47]))

                data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
                )

                train_idg = SlidingFrameGenerator(
                    classes=constants.LABELS[38:47],
                    glob_pattern=constants.MPI_WONE_AUGMENTED_DATASET_PATH + '/{classname}/*.avi',
                    nb_frames=time_steps,
                    split_val=.2,
                    shuffle=True,
                    batch_size=batch_size,
                    target_shape=(224, 224),
                    nb_channel=3,
                    transformation=data_aug,
                    use_frame_cache=False
                )

                utils.save_sample(f'{path}/sample.png', train_idg)

                validation_idg = train_idg.get_validation_generator()

                row = {
                    'trial': f'paper_01_trial_{TRIAL}',
                    'cycle': 'training',
                    'code': code,
                    'batch_size': batch_size,
                    'time_steps': time_steps,
                    'files': train_idg.files_count,
                    'sequences': len(train_idg.vid_info)
                }
                data = data.append(row, ignore_index=True)

                row = {
                    'trial': f'paper_01_trial_{TRIAL}',
                    'cycle': 'validation',
                    'code': code,
                    'batch_size': batch_size,
                    'time_steps': time_steps,
                    'files': validation_idg.files_count,
                    'sequences': len(validation_idg.vid_info)
                }
                data = data.append(row, ignore_index=True)

                # Configure callbacks
                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=path + '/model',
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True,
                        verbose=1,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=int(EPOCHS * .02)
                    ),
                    tf.keras.callbacks.CSVLogger(
                        filename=path + '/log.csv'
                    ),
                    # Avoids the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2
                    # GeForce RTX 2080 SUPER GPU cards.
                    # tf.keras.callbacks.TensorBoard(
                    #     log_dir=path + '/tb',
                    #     histogram_freq=1
                    # ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        verbose=1
                    ),
                ]

                history = model.fit(
                    train_idg,
                    validation_data=validation_idg,
                    callbacks=callbacks,
                    epochs=EPOCHS,
                )

                utils.plot_acc_loss(history, path + '/plot.png')

    data.to_csv(TRL_PATH + '/sequences.csv')


if __name__ == '__main__':
    train()
