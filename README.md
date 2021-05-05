# expert-robot
An attempt to build a CNN-LTSM classifier to learn facial emotional expressions from MPI Large Database[1]:

* smiling_encouraging
* smiling_endearment
* smiling_flirting
* smiling_sad-nostalgia
* smiling_sardonic
* smiling_triumphant
* smiling_uncertain
* smiling_winning
* smiling_yeah-right

## Requirements
* Python 3.8
* TensorFlow 2.2.0

## Download the database
```
PYTHONPATH=$PYTHONPATH:. python util/download_large_mpi_db.py
```

## Build videos
```
PYTHONPATH=$PYTHONPATH:. python util/make_videos.py
```

## Build augmented videos
```
PYTHONPATH=$PYTHONPATH:. python util/make_augmented_videos.py
```

[1] K. Kaulard, D. W. Cunningham, H. H. Bülthoff, and C. Wallraven, “The MPI Facial Expression Database — A Validated Database of Emotional and Conversational Facial Expressions,” PLoS ONE, vol. 7, no. 3, p. e32321, Mar. 2012, doi: 10.1371/journal.pone.0032321.


