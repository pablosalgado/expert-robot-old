# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#

import logging
import os
import pathlib

import numpy as np
import pandas
import tensorflow as tf
from sklearn import metrics

from common import constants, utils
from generators.OverlappedSequence import OverlappedSequenceBuilder
# Avoid the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2 GeForce RTX 2080
# SUPER GPU cards.
from generators.OverlappedSlidingWindow import OverlappedSlidingWindow

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


def evaluate():
    MODELS_PATH = pathlib.Path(__file__).parent / 'models'
    trial = 5
    code = 'islf'
    batch_size = 32
    sequence_size = 12
    overlap = 0.8
    path = f'trial_{trial:02}/{code}/{batch_size}/{sequence_size}/{overlap}'
    model_path = MODELS_PATH / path / 'model'

    # Load the best saved model.
    tf.config.experimental_run_functions_eagerly(True)

    cnn_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    # tf.keras.utils.plot_model(cnn_model, "cnn_model.png", show_shapes=True)

    rnn_model = tf.keras.models.load_model(model_path)

    # tf.keras.utils.plot_model(rnn_model, "rnn_model.png", show_shapes=True)

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.TimeDistributed(
            cnn_model,
            input_shape=(12, 224, 224, 3)
        )
    )
    model.add(rnn_model)

    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
    )

    x = OverlappedSlidingWindow(
        overlap=overlap,
        classes=LABELS,
        glob_pattern=constants.KERAS_PATH + '/evaluation/' + constants.MPI_WONE_AUGMENTED_DATASET + '/{classname}/*.avi',
        nb_frames=sequence_size,
        split_val=None,
        shuffle=False,
        batch_size=1,
        target_shape=(224, 224),
        nb_channel=3,
        transformation=data_aug,
        use_frame_cache=False
    )

    # --------------------------------------------------------------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------------------------------------------------------------
    print(f'Evaluating: {model_path}')
    evaluation = model.evaluate(x, verbose=1)
    print(evaluation)

    # --------------------------------------------------------------------------------------------------------------
    # Build the confusion matrix and the classification report for the current model.
    # --------------------------------------------------------------------------------------------------------------

    # Build the array of actual class for each generated sequence.
    labels = [LABELS.index(c) for c in [pathlib.PurePath(vi['name']).parts[-2] for vi in x.vid_info]]

    # Make predictions.
    model_predictions = model.predict(x, verbose=1)

    # Build the array of predicted classes by taking the class with the highest probability.
    predictions = [prediction.argmax() for prediction in model_predictions]

    # Build and save the confusion matrix.
    cm = tf.math.confusion_matrix(labels, predictions=predictions)
    cm_filename = f'{"cm_"}{path.replace("/", "_")}.csv'
    np.savetxt(cm_filename, cm.numpy(), delimiter=',')

    # Build the classification report
    cr_dict = metrics.classification_report(
        labels,
        predictions,
        target_names=labels,
        output_dict=True
    )

    print(cr_dict)


if __name__ == '__main__':
    os.makedirs(f'{TRL_PATH}', exist_ok=True)
    logging.basicConfig(filename=f'{TRL_PATH}/error.log', filemode='w')
    logger = logging.getLogger(__name__)
    try:
        evaluate()
    except Exception:
        logger.exception("")
        raise
