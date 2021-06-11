import logging
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

from common import constants, utils
from generators.OverlappedSlidingWindow import OverlappedSlidingWindow

TRIALS = [n for n in np.arange(1, 11, 1)]
CODES = constants.CODES[:]
BATCH_SIZES = [16, 32, 64]
TIME_STEPS = [6, 12, 24, 48]  # The smallest video just have 56 frames
OVERLAPS = [round(n, 1) for n in np.arange(0, 1, 0.1)]

MODELS_PATH = pathlib.Path(__file__).parent.parent / 'models'
EVALUATIONS_FILE = pathlib.Path(__file__).parent / 'evaluation.csv'
ERRORS_FILE = pathlib.Path(__file__).parent / 'error.log'
CLASSES = constants.LABELS[38:47]

# Avoid the CUPTI_ERROR_INSUFFICIENT_PRIVILEGES running in the ATCBIOSIMUL server at UGR with 2 GeForce RTX 2080
# SUPER GPU cards.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


class Evaluation:
    def __init__(self) -> None:
        super().__init__()

        self._evaluations = pd.DataFrame(
            columns=[
                'trial',
                'code',
                'batch_size',
                'sequence_size',
                'overlap',
                'epoch',
                'accuracy',
                'loss',
                'val_accuracy',
                'val_loss',
                'test_accuracy',
                'test_loss'
            ]
        )

    def evaluate_model(self, trial, code, batch_size, sequence_size, overlap=None):
        if overlap is None:
            path = f'trial_{trial:02}/{code}/{batch_size}/{sequence_size}'
        else:
            path = f'trial_{trial:02}/{code}/{batch_size}/{sequence_size}/{overlap}'

        model_path = MODELS_PATH / path / 'model'
        log_path = MODELS_PATH / path / 'log.csv'

        if not model_path.exists() or not log_path.exists():
            return

        log_data = pd.read_csv(log_path.as_posix())
        max_val_accuracy = log_data.iloc[log_data.val_accuracy.idxmax()]

        utils.prepare_test_dataset(constants.MPI_WONE_AUGMENTED_DATASET, 'evaluation', code)

        tf.config.experimental_run_functions_eagerly(True)
        model = tf.keras.models.load_model(model_path)

        data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
        )

        if 1 == trial:
            x = SlidingFrameGenerator(
                classes=CLASSES,
                glob_pattern=constants.KERAS_PATH + '/evaluation/' + constants.MPI_WONE_AUGMENTED_DATASET + '/{classname}/*.avi',
                nb_frames=sequence_size,
                split_val=.2,
                shuffle=True,
                batch_size=batch_size,
                target_shape=(224, 224),
                nb_channel=3,
                transformation=data_aug,
                use_frame_cache=False
            )
        else:
            x = OverlappedSlidingWindow(
                overlap=overlap,
                classes=CLASSES,
                glob_pattern=constants.KERAS_PATH + '/evaluation/' + constants.MPI_WONE_AUGMENTED_DATASET + '/{classname}/*.avi',
                nb_frames=sequence_size,
                split_val=.2,
                shuffle=True,
                batch_size=batch_size,
                target_shape=(224, 224),
                nb_channel=3,
                transformation=data_aug,
                use_frame_cache=False
            )

        print(f'Evaluating: {model_path}')
        evaluation = model.evaluate(x, verbose=1)

        self._evaluations = self._evaluations.append(
            {
                'trial': f'trial_{trial:02}',
                'code': code,
                'batch_size': batch_size,
                'sequence_size': sequence_size,
                'overlap': overlap,
                'epoch': max_val_accuracy.epoch,
                'accuracy': max_val_accuracy.accuracy,
                'loss': max_val_accuracy.loss,
                'val_accuracy': max_val_accuracy.val_accuracy,
                'val_loss': max_val_accuracy.val_loss,
                'test_accuracy': evaluation[1],
                'test_loss': evaluation[0]
            },
            ignore_index=True
        )

        self._evaluations.to_csv(EVALUATIONS_FILE.as_posix())
        print(evaluation)

    def evaluate_time_step(self, trial, code, batch_size, time_step):
        if 1 == trial:
            self.evaluate_model(trial, code, batch_size, time_step)
        else:
            for overlap in OVERLAPS:
                self.evaluate_model(trial, code, batch_size, time_step, overlap)

    def evaluate_batch(self, trial, code, batch_size):
        for time_step in TIME_STEPS:
            self.evaluate_time_step(trial, code, batch_size, time_step)

    def evaluate_code(self, trial, code):
        for batch_size in BATCH_SIZES:
            self.evaluate_batch(trial, code, batch_size)

    def evaluate_trial(self, trial):
        for code in CODES:
            self.evaluate_code(trial, code)

    def evaluate(self):
        for trial in TRIALS:
            self.evaluate_trial(trial)


if __name__ == '__main__':
    logging.basicConfig(filename=f'{ERRORS_FILE.as_posix()}', filemode='w')
    logger = logging.getLogger(__name__)
    try:
        Evaluation().evaluate()
    except Exception:
        print("Ended with errors.")
        logger.exception("")
