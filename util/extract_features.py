import pathlib
import shutil

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import constants

CODES = constants.CODES
LABELS = constants.LABELS[38:47]

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def extract_features() -> None:
    # Remove all of the extracted directories from the dataset and extract them again.
    shutil.rmtree(f'{constants.KERAS_PATH}/{constants.MPI_WONE_AUGMENTED_DATASET}', ignore_errors=True)
    tf.keras.utils.get_file(
        fname=f'{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
        origin=f'https://s3.us-east-2.amazonaws.com/datasets.pablosalgado.co/lg_mpi_db/{constants.MPI_WONE_AUGMENTED_DATASET}.tar',
        extract=True
    )

    model = tf.keras.applications.MobileNet(include_top=False)

    for label in LABELS:
        for code in CODES:
            for c in range(0, 20):
                features = []
                frames = []

                # Open video
                filename = pathlib.Path(
                    f'{constants.KERAS_DATASETS_PATH}/{constants.MPI_WONE_AUGMENTED_DATASET}/{label}/{code}_{label}_{c:02}.avi').as_posix()
                cap = cv2.VideoCapture(filename)
                while True:
                    # Grab the next frame
                    grabbed, frame = cap.read()
                    if not grabbed:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # plt.imshow(frame)
                    # plt.show()

                    frames.append(frame)

                cap.release()

                # Extract features
                x = tf.keras.applications.mobilenet.preprocess_input(
                    np.array(frames)
                )
                tf.keras.backend.clear_session()
                features = model.predict(
                    x,
                    verbose=1
                )

                # And save to a dataset
                file = h5py.File('expert-robot.h5', 'a')

                print(f'Saving: {filename}')
                group_name = f'/features/224x224/wone/augmented/mobilenet/conv_pw_13_relu/{label}'
                if group_name not in file:
                    group = file.create_group(
                        group_name
                    )
                else:
                    group = file[group_name]

                dataset_name = f'{code}_{label}_{c:02}'
                if dataset_name not in group:
                    dataset = group.create_dataset(
                        dataset_name,
                        data=features,
                        compression='gzip',
                        compression_opts=9
                    )

                file.close()


if __name__ == '__main__':
    extract_features()
