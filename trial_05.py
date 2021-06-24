# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#

import glob
import logging
import os
import shutil

import pandas
import tensorflow as tf

from common import constants, utils
from generators.OverlappedSlidingWindow import OverlappedSlidingWindow

# Avoid the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2 GeForce RTX 2080
# SUPER GPU cards.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parameters
TRIAL = 'trial_05'
CODES = constants.CODES[:]
BATCH_SIZE = [32]
TIME_STEPS = [12]  # The smallest video just have 56 frames
OVERLAPS = [0.8]
EPOCHS = 100
PATIENCE = 10

TRL_PATH = f'models/{TRIAL}'


def build_model(time_steps, nout):
    # cnn_model = tf.keras.applications.mobilenet.MobileNet()
    # cnn_model.summary();
    # tf.keras.backend.clear_session()

    # Load MobileNet model excluding top.
    cnn_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    # cnn_model.summary();
    # tf.keras.backend.clear_session()

    # Freeze all layers.
    for layer in cnn_model.layers:
        layer.trainable = False

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
    data = pandas.DataFrame(None, columns=['trial', 'cycle', 'code', 'batch_size', 'time_steps', 'overlap', 'files',
                                           'sequences'])
    data['trial'] = TRIAL

    for code in CODES:
        # Remove all of the extracted directories from the dataset and extract them again.
        shutil.rmtree(f'{constants.KERAS_PATH}/{TRIAL}/{constants.MPI_WONE_AUGMENTED_DATASET}', ignore_errors=True)
        tf.keras.utils.get_file(
            fname=f'{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
            origin=f'https://s3.us-east-2.amazonaws.com/datasets.pablosalgado.co/lg_mpi_db/{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
            cache_subdir=TRIAL,
            extract=True
        )

        # Delete the next actress to leave her out of the training.
        files = glob.glob(f'{constants.KERAS_PATH}/{TRIAL}/{constants.MPI_WONE_AUGMENTED_DATASET}/**/{code}*.avi',
                          recursive=True)
        for file in files:
            os.remove(file)

        for batch_size in BATCH_SIZE:
            for time_steps in TIME_STEPS:
                for overlap in OVERLAPS:
                    tf.keras.backend.clear_session()

                    path = TRL_PATH + f'/{code}/{batch_size}/{time_steps}/{overlap}'
                    os.makedirs(path, exist_ok=True)

                    # Build and compile the model.
                    model = build_model(time_steps, len(constants.LABELS[38:47]))

                    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
                    )

                    train_idg = OverlappedSlidingWindow(
                        overlap=overlap,
                        classes=constants.LABELS[38:47],
                        glob_pattern=constants.KERAS_PATH + '/' + TRIAL + '/' + constants.MPI_WONE_AUGMENTED_DATASET + '/{classname}/*.avi',
                        nb_frames=time_steps,
                        split_val=.2,
                        shuffle=True,
                        batch_size=batch_size,
                        target_shape=(224, 224),
                        nb_channel=3,
                        transformation=data_aug,
                        use_frame_cache=False
                    )

                    sample_path = TRL_PATH + f'/{code}/{batch_size}/{time_steps}/sample.png'
                    if not os.path.exists(sample_path):
                        utils.save_sample(sample_path, train_idg)

                    validation_idg = train_idg.get_validation_generator()

                    row = {
                        'trial': f'{TRIAL}',
                        'cycle': 'training',
                        'code': code,
                        'batch_size': batch_size,
                        'time_steps': time_steps,
                        'overlap': overlap,
                        'files': train_idg.files_count,
                        'sequences': len(train_idg.vid_info)
                    }
                    data = data.append(row, ignore_index=True)

                    row = {
                        'trial': f'{TRIAL}',
                        'cycle': 'validation',
                        'code': code,
                        'batch_size': batch_size,
                        'time_steps': time_steps,
                        'overlap': overlap,
                        'files': validation_idg.files_count,
                        'sequences': len(validation_idg.vid_info)
                    }
                    data = data.append(row, ignore_index=True)

                    data.to_csv(TRL_PATH + '/sequences.csv')

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
                            patience=PATIENCE
                        ),
                        tf.keras.callbacks.CSVLogger(
                            filename=path + '/log.csv'
                        ),
                        # Avoid the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2
                        # GeForce RTX 2080 SUPER GPU cards.
                        # tf.keras.callbacks.TensorBoard(
                        #     log_dir=path + '/tb',
                        #     histogram_freq=1
                        # ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            verbose=1,
                            patience=PATIENCE
                        ),
                    ]

                    history = model.fit(
                        train_idg,
                        validation_data=validation_idg,
                        callbacks=callbacks,
                        epochs=EPOCHS,
                    )

                    utils.plot_acc_loss(history, f'{path}/{code}-{batch_size}-{time_steps}-{overlap}-plot.png')


if __name__ == '__main__':
    os.makedirs(f'{TRL_PATH}', exist_ok=True)
    logging.basicConfig(filename=f'{TRL_PATH}/error.log', filemode='w')
    logger = logging.getLogger(__name__)
    try:
        train()
    except Exception:
        logger.exception("")
        raise
