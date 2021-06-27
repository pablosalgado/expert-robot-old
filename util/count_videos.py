import pathlib

import cv2
import pandas
import glob
import numpy as np

from common import constants

pathlib.Path(constants.LG_MPI_DB_PATH)


def count_videos():
    csv_file = f'{pathlib.Path(__file__).parent.as_posix()}/videos.csv'

    df = pandas.read_csv(
        csv_file
    )
    # df.astype(
    #     {
    #         'code': str,
    #         'label': str,
    #         'start': int,
    #         'stop': int,
    #         'frames': int
    #     }
    # )

    for index, row in df.iterrows():
        count = len(
            glob.glob(
                f'{constants.LG_MPI_DB_PATH}/**/{row.code}_{row.label}*.png'
            )
        )

        print(f'{row.code}, {row.label}, {count}')

        df.loc[index, 'frames'] = count

    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    count_videos()
