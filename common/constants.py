import pathlib

# Current user home directory
HOME = str(pathlib.Path.home())

# Directory to download the Large MPI DB
LARGE_MPI_DB_PATH = f'{HOME}/.keras/large_mpi_db'

# Videos without neutral expressions
VIDEOS_WONE = f'{HOME}/.keras/large_mpi_videos_without_neutral_expressions'

# The Large MPI DB were recorded with 10 actors and actresses. These are their
# codes.
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
