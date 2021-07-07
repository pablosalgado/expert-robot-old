# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#

import logging
import os

import pandas
import tensorflow as tf

from common import constants, utils
from generators.OverlappedSequence import OverlappedSequenceBuilder

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
LABELS = constants.LABELS[38:47]
EPOCHS = 100
PATIENCE = 10

TRL_PATH = f'models/{TRIAL}'


def build_model(time_steps, nout):
    rnn_model = tf.keras.models.Sequential()

    rnn_model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAvgPool2D(),
            input_shape=(time_steps, 7, 7, 1024)
        )
    )

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

                    builder = OverlappedSequenceBuilder(
                        filename='/home/psalgado/expert-robot/expert-robot.h5',
                        path='/features/224x224/wone/augmented/mobilenet/conv_pw_13_relu',
                        labels=LABELS,
                        overlap=overlap,
                        sequence_size=time_steps,
                        split=0.8,
                        test=code
                    )

                    row = {
                        'trial': f'{TRIAL}',
                        'cycle': 'training',
                        'code': code,
                        'batch_size': batch_size,
                        'time_steps': time_steps,
                        'overlap': overlap,
                        'files': builder.files_count,
                        'sequences': builder.get_training_sequence().get_sequences_count()
                    }
                    data = data.append(row, ignore_index=True)

                    row = {
                        'trial': f'{TRIAL}',
                        'cycle': 'validation',
                        'code': code,
                        'batch_size': batch_size,
                        'time_steps': time_steps,
                        'overlap': overlap,
                        'files': builder.files_count,
                        'sequences': builder.get_validation_sequence().get_sequences_count()
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
                        builder.get_training_sequence(),
                        validation_data=builder.get_validation_sequence(),
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
