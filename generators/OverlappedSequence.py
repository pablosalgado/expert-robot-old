import tensorflow as tf
import h5py
import numpy as np


class OverlappedSequence(tf.keras.utils.Sequence):
    def __init__(
            self,
            labels,
            overlap,
            batch_size=32,
            sequence_size=6,
            shuffle=True
    ) -> None:
        super().__init__()

        self.__labels = labels
        self.__overlap = overlap
        self.__batch_size = batch_size
        self.__sequence_size = sequence_size
        self.__shuffle = shuffle

        self.__filename = '/home/psalgado/expert-robot/expert-robot.h5'
        self.__sequences = []
        self.__indexes = np.arange(0)

        self.__init_sequences()
        self.on_epoch_end()

    def __len__(self):
        len(self.__sequences) / self.__batch_size

    def __getitem__(self, index):
        pass

    def on_epoch_end(self):
        self.__indexes = np.arange(len(self.__sequences))

        if self.__shuffle:
            np.random.shuffle(self.__indexes)

    def __init_sequences(self) -> None:
        file = h5py.File(self.__filename, 'r')

        path = '/features/224x224/wone/augmented/mobilenet/conv_pw_13_relu'

        files_count = 0

        print('Checking files to find possible sequences, please wait...')

        for l1 in file[path].items():
            label = l1[0]
            items = l1[1].items()

            for l2 in items:
                files_count += 1

                # Dataset has 1 tensor of extracted features for each frame from the original video,
                # so if the video had 50 frames this tensor is shape (50, 7, 7, 1024) if the feature
                # extraction was achieved with a MobileNet without top.
                dataset = l2[1]


                # Calculate all possible sequences
                i = 0
                while i + self.__sequence_size <= len(dataset):
                    self.__sequences.append(
                        {
                            'label': label,
                            'frames': np.arange(i, i + self.__sequence_size),
                        }
                    )

                    i = round(i + self.__sequence_size - self.__sequence_size * self.__overlap)

        print(f'For {files_count} files, I found {len(self.__sequences)} possible sequence samples')

        self.__indexes = np.arange(
            len(self.__sequences)
        )

        file.close()
