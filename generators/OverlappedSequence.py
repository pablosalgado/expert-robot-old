import tensorflow as tf
import h5py
import numpy as np


class OverlappedSequenceBuilder:
    """
    A builder of overlapped sequences of feature vectors extracted from video frames from an HDF5 file. A path to the
    datasets must be provided. The datasets are expected to be grouped by labels. Each dataset is comprised as a set of
    the extracted feature vector from each video frame. A sequence es then generated as a subset of a given dataset
    according to the provided sequence size.    
    """

    def __init__(
            self,
            filename: str,
            path: str,
            labels: list,
            overlap: float,
            sequence_size: int,
            split: float,
            batch_size: int = 32,
            shuffle=True
    ) -> None:
        super().__init__()

        self.__filename = filename
        self.__path = path
        self.__labels = labels
        self.__overlap = overlap
        self.__sequence_size = sequence_size
        self.__split = split
        self.__batch_size = batch_size
        self.__shuffle = shuffle

        # Make sure labels are sorted alphabetically
        self.__labels.sort()

        self.__sequences = []
        self.__training_sequences = []
        self.__validation_sequences = []

        self.__find_sequences()

        self.__split_sequences()

    def __find_sequences(self) -> None:
        """
        Find all possible sequences of size self.__sequence_size for all feature datasets.
        :return: None
        """
        file = h5py.File(self.__filename, 'r')

        files_count = 0

        print('Checking files to find possible sequences, please wait...')

        for label in self.__labels:
            names = file[self.__path][label]

            for name in names:
                files_count += 1

                dataset = file[self.__path][label][name]

                # Calculate all possible sequences
                i = 0
                while i + self.__sequence_size <= len(dataset):
                    self.__sequences.append(
                        {
                            'label': label,
                            'name': name,
                            'frames': np.arange(i, i + self.__sequence_size),
                        }
                    )

                    i = round(i + self.__sequence_size - self.__sequence_size * self.__overlap)

        print(f'For {files_count} files, I found {len(self.__sequences)} possible sequence samples')

        file.close()

    def __split_sequences(self):
        indexes = np.arange(
            len(self.__sequences)
        )

        if self.__shuffle:
            np.random.shuffle(indexes)

        split = int(
            self.__split * len(indexes)
        )

        t_idxs = indexes[:split]
        v_idxs = indexes[split:]

        self.__training_sequences = [self.__sequences[i] for i in t_idxs]
        self.__validation_sequences = [self.__sequences[i] for i in v_idxs]

    def get_training_sequence(self):
        return OverlappedSequence(
            filename=self.__filename,
            path=self.__path,
            sequences=self.__training_sequences,
            labels=self.__labels,
            overlap=self.__overlap,
            sequence_size=self.__sequence_size,
            batch_size=self.__batch_size,
            shuffle=self.__shuffle
        )

    def get_validation_sequence(self):
        return OverlappedSequence(
            filename=self.__filename,
            path=self.__path,
            sequences=self.__validation_sequences,
            labels=self.__labels,
            overlap=self.__overlap,
            sequence_size=self.__sequence_size,
            batch_size=self.__batch_size,
            shuffle=self.__shuffle
        )


class OverlappedSequence(tf.keras.utils.Sequence):

    def __init__(
            self,
            filename: str,
            path: str,
            sequences: list,
            labels: list,
            overlap: float,
            sequence_size: int,
            batch_size: int = 32,
            shuffle=True
    ) -> None:
        super().__init__()

        self.__filename = filename
        self.__path = path
        self.__sequences = sequences
        self.__labels = labels
        self.__overlap = overlap
        self.__sequence_size = sequence_size
        self.__batch_size = batch_size
        self.__shuffle = shuffle

        self.__indexes = None

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.floor(
                len(self.__sequences) / self.__batch_size
            )
        )

    def __getitem__(self, index):
        file = h5py.File(self.__filename, 'r')

        indexes = self.__indexes[index * self.__batch_size:(index + 1) * self.__batch_size]

        batch = []
        labels = []
        for i in indexes:
            sequence = self.__sequences[i]

            dataset = file[f'{self.__path}/{sequence["label"]}/{sequence["name"]}']

            batch.append(
                np.array(
                    dataset[sequence['frames'][0]:sequence['frames'][-1] + 1]
                )
            )

            label = np.zeros(
                len(self.__labels)
            )
            label[
                self.__labels.index(
                    sequence['label']
                )
            ] = 1

            labels.append(label)

        file.close

        return np.array(batch), np.array(labels)

    def on_epoch_end(self):
        self.__indexes = np.arange(
            len(self.__sequences)
        )

        if self.__shuffle:
            np.random.shuffle(self.__indexes)
