# -----------------------------------------------------------------------------
# Builds videos from Large MPI Facial Expression Database pictures.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
# -----------------------------------------------------------------------------

import pathlib

import cv2
import pandas

from common import constants


def without_neutral():
    """
    Builds a set of videos excluding neutral facial expression frames.

    The resulting dataset is available at:
    https://datasets.pablosalgado.co/mpi/large/videos/large-mpi-videos-without-neutral-expressions.tar.gz

    :return: None
    """

    df = pandas.read_csv('make_videos.csv')

    for label in constants.LABELS[38:47]:
        for code in constants.CODES:
            q = df.query(f'code == "{code}" & label == "{label}"')
            start = q.iloc[0].start
            stop = q.iloc[0].stop + 1
            picture_array = []
            for index in range(start, stop):
                template_path = f'{constants.LARGE_MPI_DB_PATH}/{label}/{code}_{label}_{index:03}.png'
                print(template_path)
                picture_path = pathlib.Path(template_path).absolute().as_posix()
                picture = cv2.imread(picture_path)
                height, width, _ = picture.shape
                size = (width, height)
                picture_array.append(picture)

            # Write video
            pathlib.Path(f'{constants.VIDEOS_WONE}/{label}').mkdir(parents=True, exist_ok=True)
            video_path = pathlib.Path(f'{constants.VIDEOS_WONE}/{label}/{code}_{label}.avi')

            print(video_path.absolute().as_posix())

            out = cv2.VideoWriter(
                video_path.absolute().as_posix(),
                cv2.VideoWriter_fourcc(*'DIVX'),
                25,
                size
            )

            for i in range(len(picture_array)):
                out.write(picture_array[i])

            out.release()


if __name__ == '__main__':
    without_neutral()
