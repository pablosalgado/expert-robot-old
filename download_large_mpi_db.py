# -----------------------------------------------------------------------------
# Download "Large MPI Facial Expression Database".
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# This DB is huge, about 30GB. Although one file is available for download, this
# script instead downloads 130 compressed files.
#
# Each file is downloaded and decompressed in "~/.keras/large-mpi-db"
# Once all files are downloaded and decompressed, 51 directories are created,
# each named after the conversational expression found in each video.
#
# This DB provides pictures for the n frames of each video not the videos
# themselves, so each frame is numbered:
#
# ~/.keras/
#   large-mpi-db/
#     agree_considered/
#       cawm_agree_considered_001.png
#       cawm_agree_considered_002.png
#       cawm_agree_considered_003.png
#     agree_continue/
#       cawm_agree_continue_001.png
#       cawm_agree_continue_002.png
#       cawm_agree_continue_003.png
import pathlib
import re

import tensorflow as tf

from common import constants

URL_PREFIX = 'http://www.informatik.tu-cottbus.de/gs/ZipArchiveLargeDB/1_ZipArchive-CentralCam_old-MPI-Parsing/'


def download():
    # Ten actors and actresses were recorded, each part is named after them.
    for n, code in enumerate(constants.CODES, start=1):
        # 13 parts are provided for each actor or actress.
        for part in range(1, 14):
            # Build filename.
            filename = f'MPI_large_centralcam_hi_{code}_{n:02}-{part:02}.zip'

            # Build download url.
            url = f'{URL_PREFIX}{n:02}_{code}/{filename}'

            print(url)

            # Let Keras take care of downloading and decompressing.
            tf.keras.utils.get_file(
                fname=filename,
                origin=url,
                extract=True,
                cache_subdir='large_mpi_db'
            )


def normalize():
    """
    Normalize the extracted directories and files by changing '-' to '_' and case to lower.
    :return: Nothing
    """
    for path in pathlib.Path(f'{constants.LARGE_MPI_DB_PATH}').iterdir():
        if path.is_dir():
            # Directory name will be the label of the class
            label = path.name.lower().replace('-', '_')

            for file in path.glob('*'):
                # Changes current filename to lower case and replaces '-' by '_'
                filename = file.name.lower().replace('-', '_')

                # Fix the filename replacing the label part
                filename = re.sub(
                    r'((cawm|chsm|islf|jakm|juhm|kabf|lekf|mamm|milf|silf)_).+(_[0-9]{3}\.png)',
                    f'\\1{label}\\3',
                    filename
                )

                # Rename the file
                old = file.as_posix()
                new = file.rename(
                    pathlib.Path(file.parent, filename)
                ).as_posix()
                print(f'{old} -> {new}')

            # Rename the directory with label
            old = path.as_posix()
            new = path.rename(
                pathlib.Path(path.parent, label)
            ).as_posix()
            print(f'{old} -> {new}')


def delete():
    for path in pathlib.Path(f'{constants.LARGE_MPI_DB_PATH}').iterdir():
        if path.is_dir():
            for file in path.glob('*'):
                file.unlink()
            path.rmdir()


if __name__ == "__main__":
    delete()
    download()
    normalize()
