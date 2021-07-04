import tensorflow as tf
import h5py
import numpy as np


class OverlappedSequence(tf.keras.utils.Sequence):
    def __init__(
            self,
            labels: list = None,
            overlap: float = None,
            batch_size: int = 32,
            sequence_size: int = None,
            shuffle=True
    ) -> None:
        super().__init__()

        self.__labels = labels
        self.__overlap = overlap
        self.__batch_size = batch_size
        self.__sequence_size = sequence_size
        self.__shuffle = shuffle

        # Make sure label are sorted alphabetically
        self.__labels.sort()

        self.__filename = '/home/psalgado/expert-robot/expert-robot.h5'
        self.__path = '/features/224x224/wone/augmented/mobilenet/conv_pw_13_relu'
        self.__seq_info = []
        self.__indexes = np.arange(0)

        self.__init_sequences()
        self.on_epoch_end()

    def __len__(self):
        return int(
            np.floor(
                len(self.__seq_info) / self.__batch_size
            )
        )

    def __getitem__(self, index):
        file = h5py.File(self.__filename, 'r')

        indexes = self.__indexes[index * self.__batch_size:(index + 1) * self.__batch_size]

        batch = []
        labels = []
        for i in indexes:
            seq_info = self.__seq_info[self.__indexes[i]]

            dataset = file[f'{self.__path}/{seq_info["label"]}/{seq_info["name"]}']

            batch.append(
                np.array(
                    dataset[seq_info['frames'][0]:seq_info['frames'][-1] + 1]
                )
            )

            label = np.zeros(
                len(self.__labels)
            )
            label[
                self.__labels.index(
                    seq_info['label']
                )
            ] = 1

            labels.append(label)

        file.close

        return np.array(batch), np.array(labels)

    def on_epoch_end(self):
        self.__indexes = np.arange(
            len(self.__seq_info)
        )

        if self.__shuffle:
            np.random.shuffle(self.__indexes)

    def __init_sequences(self) -> None:
        file = h5py.File(self.__filename, 'r')

        files_count = 0

        print('Checking files to find possible sequences, please wait...')

        for l1 in file[self.__path].items():
            label = l1[0]
            items = l1[1].items()

            for l2 in items:
                files_count += 1

                name = l2[0]
                dataset = l2[1]

                # Calculate all possible sequences
                i = 0
                while i + self.__sequence_size <= len(dataset):
                    self.__seq_info.append(
                        {
                            'label': label,
                            'name': name,
                            'frames': np.arange(i, i + self.__sequence_size),
                        }
                    )

                    i = round(i + self.__sequence_size - self.__sequence_size * self.__overlap)

        print(f'For {files_count} files, I found {len(self.__seq_info)} possible sequence samples')

        self.__indexes = np.arange(
            len(self.__seq_info)
        )

        file.close()
