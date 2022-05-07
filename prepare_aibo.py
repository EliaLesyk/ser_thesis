import pandas as pd
import pickle
import os
from os.path import isfile

DATA_DIR = "/Users/el/embrace/data/"

AIBO_PATH = DATA_DIR + 'AIBO/aibo.pkl'
aibo_labels_file = "AIBO/labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt"

# Anger (angry, touchy, and reprimanding), Emphatic, Neutral, Positive (motherese and joyful),and Rest
aibo_dict = {'N': 'neutral', 'E': 'empathic', 'A': 'angry', 'R': 'rest', 'P': 'positive'}


def read_aibo():
    if not isfile(AIBO_PATH):
        print("AIBO not found. Creating AIBO dataset...")
        df = pd.read_csv(os.path.join(DATA_DIR, aibo_labels_file), sep=" ", header=None,
                         names=["filename", "label", "percentage"])
        with open(AIBO_PATH, "wb") as file:
            pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)
    iemocap = pickle.load(open(AIBO_PATH, "rb"))
    return iemocap
