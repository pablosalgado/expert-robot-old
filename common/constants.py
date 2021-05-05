import pathlib

# Current user home directory
HOME = str(pathlib.Path.home())

LG_MPI_DB='lg_mpi_db'

# Directory to download the Large MPI DB
LG_MPI_DB_PATH = f'{HOME}/.keras/{LG_MPI_DB}'

# MPI videos
VIDEOS = f'{HOME}/.keras/lg_mpi_videos'

# MPI videos without neutral expressions
WONE_VIDEOS = f'{HOME}/.keras/lg_mpi_wone_videos'
WONE_AUGMENTED_VIDEOS = f'{HOME}/.keras/lg_mpi_wone_augmented_videos'

# The Large MPI DB were recorded with 10 actors and actresses. These are their
# codes. Don't sort as this is the order to download.
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')

# The class names or categories for each facial expression.
LABELS = [
    'agree_considered',
    'agree_continue',
    'agree_pure',
    'agree_reluctant',
    'aha_light_bulb_moment',
    'annoyed_bothered',
    'annoyed_rolling_eyes',
    'arrogant',
    'bored',
    'compassion',
    'confused',
    'contempt',
    'disagree_considered',
    'disagree_pure',
    'disagree_reluctant',
    'disbelief',
    'disgust',
    'embarrassment',
    'fear_oops',
    'fear_terror',
    'happy_achievement',
    'happy_laughing',
    'happy_satiated',
    'happy_schadenfreude',
    'i_did_not_hear',
    'i_dont_care',
    'i_dont_know',
    'i_dont_understand',
    'imagine_negative',
    'imagine_positive',
    'impressed',
    'insecurity',
    'not_convinced',
    'pain_felt',
    'pain_seen',
    'remember_negative',
    'remember_positive',
    'sad',
    'smiling_encouraging',
    'smiling_endearment',
    'smiling_flirting',
    'smiling_sad_nostalgia',
    'smiling_sardonic',
    'smiling_triumphant',
    'smiling_uncertain',
    'smiling_winning',
    'smiling_yeah_right',
    'thinking_considering',
    'thinking_problem_solving',
    'tired',
    'treudoof_bambi_eyes',
]
