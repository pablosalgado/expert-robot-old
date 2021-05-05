import os
import pathlib

import cv2
import numpy as np
import pandas
from keras_preprocessing.image.affine_transformations import apply_affine_transform, flip_axis, apply_channel_shift, \
    apply_brightness_shift

from common import constants

np.random.seed(645)


def get_random_transformations() -> []:
    """

    :return:
    """
    transformations = []

    for x in range(20):
        z = np.random.uniform(.95, 1.05)
        transformations.append({
            'theta': np.random.uniform(-5, 5),
            'tx': np.random.uniform(-5, 5),
            'ty': np.random.uniform(-5, 5),
            'shear': np.random.uniform(-5, 5),
            'zx': z,
            'zy': z,
            'flip_horizontal': np.random.uniform(0, 1) > 0.5,
            'flip_vertical': False,
            'channel_shift_intensity': None,
            'brightness': np.random.uniform(0.5, 1.5),
            'grayscale': np.random.uniform(0, 1) > 0.5,
        })

    return transformations


def augment_videos() -> None:
    """

    :return: None
    """
    df = pandas.read_csv(f'{pathlib.Path(__file__).parent.as_posix()}/videos.csv')

    for code in constants.CODES:
        for label in constants.LABELS[38:47]:

            # Applies 20 transformations
            for t_count, transformation in enumerate(get_random_transformations()):
                # Video output path
                video_path = f'{constants.WONE_AUGMENTED_VIDEOS}/{code}_{label}_{t_count:02d}.avi'

                # Checks if video has been already created
                q = df.query(f'code == "{code}" & label == "{label}"')
                start = q.iloc[0].start
                stop = q.iloc[0].stop + 1
                cap = cv2.VideoCapture(video_path)
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) == stop - start:
                    print(f'Skipping: {video_path}')
                    continue

                # Creates output dir and video output stream
                os.makedirs(constants.WONE_AUGMENTED_VIDEOS, exist_ok=True)
                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    25,
                    (224, 224)
                )

                print(f'Creating: {video_path}')

                # Applies current transformation to all frames (PNGs)
                for i_count in range(start, stop):
                    # Loads next frame (PNG)
                    template_path = f'{constants.LG_MPI_DB_PATH}/{label}/{code}_{label}_{i_count:03}.png'
                    image_path = pathlib.Path(template_path).absolute().as_posix()
                    x = cv2.imread(image_path)

                    # Resizes frame keeping aspect ratio
                    x = cv2.resize(x, (224, 168))

                    # Adds a border to fill the frame to the output size of 224x224
                    x = cv2.copyMakeBorder(x, 28, 28, 0, 0, cv2.BORDER_CONSTANT)

                    # Saves the original resized PNG for future reference
                    if i_count == 0:
                        cv2.imwrite(f'{constants.WONE_AUGMENTED_VIDEOS}/transformation_oo.png', x)

                    # Actually applies the transformation to the frame
                    x = apply_affine_transform(
                        x,
                        transformation.get('theta', 0),
                        transformation.get('tx', 0),
                        transformation.get('ty', 0),
                        transformation.get('shear', 0),
                        transformation.get('zx', 1),
                        transformation.get('zy', 1)
                    )

                    if transformation.get('channel_shift_intensity') is not None:
                        x = apply_channel_shift(
                            x,
                            transformation['channel_shift_intensity']
                        )

                    if transformation.get('flip_horizontal', False):
                        x = flip_axis(x, 1)

                    if transformation.get('flip_vertical', False):
                        x = flip_axis(x, 0)

                    if transformation.get('brightness') is not None:
                        x = apply_brightness_shift(x, transformation['brightness'])
                        x = np.uint8(x)

                    if transformation.get('grayscale', False):
                        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)

                    if i_count == 0:
                        cv2.imwrite(f'{constants.WONE_AUGMENTED_VIDEOS}/transformation_{t_count:02d}.png', x)

                    # Writes the frame to the video output stream.
                    out.write(x)

                out.release()


if __name__ == "__main__":
    augment_videos()
