import logging
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator
from sklearn import metrics

from common import constants, utils
from generators.OverlappedSlidingWindow import OverlappedSlidingWindow

TRIALS = [n for n in np.arange(1, 11, 1)]
CODES = constants.CODES[:]
BATCH_SIZES = [16, 32, 64]
TIME_STEPS = [6, 12, 24, 48]  # The smallest video just have 56 frames
OVERLAPS = [round(n, 1) for n in np.arange(0, 1, 0.1)]

CURRENT_DIR = pathlib.Path(__file__).parent
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

        if not EVALUATIONS_FILE.exists():
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
                    'test_loss',
                    'confusion_matrix',
                    'classification_report'
                ]
            )
        else:
            self._evaluations = pd.read_csv(
                EVALUATIONS_FILE.as_posix(),
                index_col=0
            )

    def evaluate_model(self, trial, code, batch_size, sequence_size, overlap=None):
        # Filter conditions to find if a given model has been evaluated.
        trial_filter = self._evaluations.trial == f'trial_{trial:02}'
        code_filter = self._evaluations.code == code
        batch_filter = self._evaluations.batch_size == batch_size
        sequence_filter = self._evaluations.sequence_size == sequence_size

        # trial_01 doesn't use the overlapped sliding window.
        if overlap is None:
            path = f'trial_{trial:02}/{code}/{batch_size}/{sequence_size}'
            overlap_filter = self._evaluations.overlap.isnull()
        else:
            path = f'trial_{trial:02}/{code}/{batch_size}/{sequence_size}/{overlap}'
            overlap_filter = self._evaluations.overlap == overlap

        model_path = MODELS_PATH / path / 'model'
        log_path = MODELS_PATH / path / 'log.csv'

        # Skip if no model has been trained.
        if not model_path.exists() or not log_path.exists():
            return

        # If a model chart is found, training is over. Otherwise skip it.
        if 0 == len([x for x in (MODELS_PATH / path).iterdir() if 'plot' in x.name]):
            return

        # Skip if no plot. Plot is drawn when training

        # Skip if model has been previously evaluated.
        found = self._evaluations[trial_filter & code_filter & batch_filter & sequence_filter & overlap_filter]
        if not found.empty:
            print(f'Skipping: {path}')
            return

        # Get training data of the saved best model.
        log_data = pd.read_csv(log_path.as_posix())
        max_val_accuracy = log_data.iloc[log_data.val_accuracy.idxmax()]

        # Prepare dataset for evaluation.
        utils.prepare_test_dataset(constants.MPI_WONE_AUGMENTED_DATASET, 'evaluation', code)

        # Load the best saved model.
        tf.config.experimental_run_functions_eagerly(True)
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception:
            # The model can be inconsistent if the training was stopped, so, simply continue with the next model.
            return

        # Build the appropriate sequence generator.
        data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
        )

        if 1 == trial:
            x = SlidingFrameGenerator(
                classes=CLASSES,
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
        else:
            x = OverlappedSlidingWindow(
                overlap=overlap,
                classes=CLASSES,
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
        labels = [CLASSES.index(c) for c in [pathlib.PurePath(vi['name']).parts[-2] for vi in x.vid_info]]

        # Make predictions.
        model_predictions = model.predict(x, verbose=1)

        # Build the array of predicted classes by taking the class with the highest probability.
        predictions = [prediction.argmax() for prediction in model_predictions]

        # Build and save the confusion matrix.
        cm = tf.math.confusion_matrix(labels, predictions=predictions)
        cm_filename = f'{"cm_"}{path.replace("/", "_")}.csv'
        np.savetxt(CURRENT_DIR / cm_filename, cm.numpy(), delimiter=',')

        # Build the classification report
        cr_dict = metrics.classification_report(
            labels,
            predictions,
            target_names=CLASSES,
            output_dict=True
        )

        classification_report = pd.DataFrame(
            columns=[
                '',
                'precision',
                'recall',
                'f1-score',
                'support',
            ]
        )

        for k in cr_dict.keys():
            if 'accuracy' == k:
                continue

            classification_report = classification_report.append(
                {
                    '': k,
                    'precision': cr_dict[k]['precision'],
                    'recall': cr_dict[k]['recall'],
                    'f1-score': cr_dict[k]['f1-score'],
                    'support': cr_dict[k]['support'],
                },
                ignore_index=True
            )

        # Save the classification report.
        cr_filename = f'{"cr_"}{path.replace("/", "_")}.csv'
        classification_report.to_csv(CURRENT_DIR / cr_filename)

        # Add the evaluation record, the confusion matrix and classification report to the evaluations track.
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
                'test_loss': evaluation[0],
                'confusion_matrix': cm_filename,
                'classification_report': cr_filename
            },
            ignore_index=True
        )

        # Finally save evaluation data to disk.
        self._evaluations.to_csv(EVALUATIONS_FILE.as_posix())

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
        raise
