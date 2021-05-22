import cv2 as cv
import numpy as np
from keras_video.generator import VideoFrameGenerator
from tensorflow.keras.preprocessing.image import img_to_array


class OverlappedSlidingWindow(VideoFrameGenerator):
    """
    SlidingFrameGenerator is useful to get several sequence of
    the same "action" by sliding the cursor of video. For example, with a
    video that have 60 frames using 30 frames per second, and if you want
    to pick 6 frames with a 50% overlap, the generator will return:

    - one sequence with frame ``[ 0,  1, 2, 3, 4, 5]``
    - then ``[ 3,  4, 5, 6, 7, 8])``
    - and so on to frame 30

    params:

    - overlap: a percentage of frames overlapped

    from VideoFrameGenerator:

    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that
        will be replaced by one of the class list
    """

    def __init__(self, *args, overlap: float = 0, **kwargs) -> None:
        super().__init__(no_epoch_at_init=True, *args, **kwargs)
        self.overlap = overlap

        self.sample_count = 0
        self.vid_info = []
        self.__frame_cache = {}
        self.__init_length()
        self.on_epoch_end()

    def __init_length(self):
        count = 0
        print("Checking files to find possible sequences, please wait...")
        for filename in self.files:
            cap = cv.VideoCapture(filename)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = self.count_frames(cap, filename)
            cap.release()

            # TODO: If a video has fewer frames than the sliding window we should create one sequence with all the
            #  frames in the video.

            i = 0
            while i + self.nbframe <= frame_count:
                self.vid_info.append({
                    'id': count,
                    'name': filename,
                    'frame_count': int(frame_count),
                    'frames': np.arange(i, i + self.nbframe),
                    'fps': fps,
                })
                count += 1
                i = round(i + self.nbframe - self.nbframe * self.overlap)

        print("For %d files, I found %d possible sequence samples" %
              (self.files_count, len(self.vid_info)))
        self.indexes = np.arange(len(self.vid_info))

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(len(self.vid_info)):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.vid_info) / self.batch_size))

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            overlap=self.overlap,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _validation_data=self.validation)

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            overlap=self.overlap,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _test_data=self.test)

    def __getitem__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            vid = self.vid_info[i]
            video = vid.get('name')
            frames = vid.get('frames')
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if vid['id'] not in self.__frame_cache:
                frames = self._get_frames(video, frames, shape)
            else:
                frames = self.__frame_cache[vid['id']]

            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_frames(self, video, indexes, shape):
        cap = cv.VideoCapture(video)

        frames = []
        frame_i = 0

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            if frame_i in indexes:
                # resize
                frame = cv.resize(frame, shape)

                # use RGB or Grayscale ?
                if self.nb_channel == 3:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                # to np
                frame = img_to_array(frame) * self.rescale

                # keep frame
                frames.append(frame)
            frame_i += 1

            if len(frames) == len(indexes):
                break

        cap.release()

        return np.array(frames)
