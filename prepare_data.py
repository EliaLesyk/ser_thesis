import pickle
from collections import Counter
import os
import tqdm
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import TensorDataset

from scipy.fftpack import dct

from preprocess_audio import generate_spectrogram, generate_mel_spectr, generate_mfcc, \
    generate_tbf, generate_ravdess_mspectr, generate_ravdess_mfcc, add_white_noise, ICP_MFCC_DUR, MFCC_STEP, lifter
from utils import timeit

DATA_DIR = "/Users/el/embrace/data/"
#DATA_DIR = "/home/yq36elyb/data/"

IEMOCAP_PATH = DATA_DIR + 'IEMOCAP/iemocap_with_noise.pkl'
RAVDESS_PATH = DATA_DIR + 'RAVDESS/ravdess.pkl'
AIBO_PATH = DATA_DIR + 'AIBO/aibo.pkl'
aibo_labels_file = "AIBO/labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt"

SPECTROGRAMS_FEATURES_PATH = DATA_DIR + "IEMOCAP/spectrograms_features.npy"
SPECTROGRAMS_LABELS_PATH = DATA_DIR + "IEMOCAP/spectrograms_labels.npy"
SPECTROGRAMS_SESSIONS_PATH = DATA_DIR + "IEMOCAP/spectrograms_sessions.npy"

MSPECTROGRAMS_FEATURES_PATH = DATA_DIR + "IEMOCAP/mspectrograms_features.npy"
MSPECTROGRAMS_LABELS_PATH = DATA_DIR + "IEMOCAP/mspectrograms_labels.npy"
MSPECTROGRAMS_SESSIONS_PATH = DATA_DIR + "IEMOCAP/mspectrograms_sessions.npy"

MFCCS_FEATURES_PATH = DATA_DIR + "IEMOCAP/mfccs_features.npy"
MFCCS_LABELS_PATH = DATA_DIR + "IEMOCAP/mfccs_labels.npy"
MFCCS_SESSIONS_PATH = DATA_DIR + "IEMOCAP/mfccs_sessions.npy"

TBF_PATH = DATA_DIR + "IEMOCAP/tbf.npy"
TBF_LABELS_PATH = DATA_DIR + "IEMOCAP/tbf_labels.npy"
TBF_SESSIONS_PATH = DATA_DIR + "IEMOCAP/tbf_sessions.npy"

RVD_AUDIOS_PATH = DATA_DIR + "RAVDESS/audios.npy"

AIBO_MFCCS_FEATURES_PATH = DATA_DIR + "AIBO/mfccs_features.npy"
AIBO_MFCCS_LABELS_PATH = DATA_DIR + "AIBO/mfccs_labels.npy"
AIBO_MFCCS_SESSIONS_PATH = DATA_DIR + "AIBO/mfccs_sessions.npy"

AIBO_MSPECTR_FEATURES_PATH = DATA_DIR + "AIBO/mspectr_features.npy"
AIBO_MSPECTR_LABELS_PATH = DATA_DIR + "AIBO/mspectr_labels.npy"
AIBO_MSPECTR_SESSIONS_PATH = DATA_DIR + "AIBO/mspectr_sessions.npy"

USED_CLASSES = ["neu", "hap", "sad", "ang", "fru", "exc"]

# USED_CLASSES = ["fru", "neu", "hap", "sad", "ang", "fea", "exc", "sur"]
#CLASS_TO_ID = {"neu": 0, "hap": 1, "sur": 2, "sad": 3, "fru": 4, "ang": 5, "fea": 6}
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "fru": 3, "ang": 4}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
ID_TO_FULL_CLASS = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Anger"}

# Anger (angry, touchy, and reprimanding), Emphatic, Neutral, Positive (motherese and joyful),and Rest
#aibo_dict = {'N': 'neutral', 'E': 'empathic', 'A': 'angry', 'R': 'rest', 'P': 'positive'}
# ICP: {"neu": 0, "hap": 1, "sad": 2, "fru": 3, "ang": 4}
# neu = N + R, ang + fru = A, hap = P

aibo_dict = dict(N=0, R=1, P=2, E=3, A=4)


def read_iemocap():
    # Extract IEMOCAP
    if not isfile(IEMOCAP_PATH):
        print("IEMOCAP not found. Creating IEMOCAP dataset...")
        labels_df = pd.read_csv(DATA_DIR + "IEMOCAP/" + "metadata.csv")
        # Create a column for gender
        labels_df['gender'] = labels_df.ID.str[-1:]
        # Convert gender to numeric
        labels_df['gender'] = labels_df['gender'].map(dict(zip(['M', 'F'], [0, 1])))
        # emotions = np.array(['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'fea', 'sur', 'dis'])
        # filt_df = labels_df[labels_df['emotion'].isin(emotions)]
        iemocap_list = labels_df[['File Name', 'emotion', 'gender']].copy()  # wav_file, label

        iemocap = []
        # keep only 5 classes
        for index, row in iemocap_list[iemocap_list['emotion'].isin(USED_CLASSES)].iterrows():
            # merge excited in happy
            if row["emotion"] == "exc":
                row["emotion"] = "hap"
            iemocap.append(row)

            # add 2 new samples with white Gaussian noise
            file_name = str(row['File Name']) + '.wav'
            session_id = file_name.split('Ses0')[1][0]
            abs_path = join(DATA_DIR, 'IEMOCAP/', "Session{}".format(session_id), file_name)
            add_white_noise(abs_path, SNR=10)
            row_noised = row.copy()     # if not .copy() all rows with 10 and 20dB noise in naming
            row_noised['File Name'] = str(row['File Name']) + "_with_10dB_noise"
            iemocap.append(row_noised)

            add_white_noise(abs_path, SNR=20)
            row_noised_ = row.copy()
            row_noised_['File Name'] = str(row['File Name']) + "_with_20dB_noise"
            iemocap.append(row_noised_)

        with open(IEMOCAP_PATH, "wb") as file:
            pickle.dump(iemocap, file, protocol=pickle.HIGHEST_PROTOCOL)
    iemocap = pickle.load(open(IEMOCAP_PATH, "rb"))
    return iemocap


def read_ravdess():
    # Extract RAVDESS
    if not isfile(RAVDESS_PATH):
        print("RAVDESS not found. Creating RAVDESS dataset...")
        ravdess_db = pd.DataFrame(columns=['path', 'source', 'actor', 'gender', 'emotion', 'emotion_lb'])
        count = 0
        #for data_path in os.path.join(DATA_DIR, "RAVDESS/data/"):
        dir_list = [i for i in os.listdir(DATA_DIR + "RAVDESS/data/") if i.startswith("Actor_")]
        dir_list.sort()
        #print("dir_list", dir_list)
        for i in dir_list:
            file_list = os.listdir(DATA_DIR + "RAVDESS/data/" + i)
            # print("file_list", file_list)
            for f in file_list:
                nm = f.split('.')[0].split('-')
                path = DATA_DIR + "RAVDESS/data/" + i + '/' + f
                #src = int(nm[1])
                actor = int(nm[-1])
                emotion = int(nm[2])
                source = "Ravdess"

                if int(actor) % 2 == 0:
                    gender = "female"
                else:
                    gender = "male"

                if emotion == 1:
                    lb = "neutral"
                elif emotion == 2:
                    lb = "calm"
                elif emotion == 3:
                    lb = "happy"
                elif emotion == 4:
                    lb = "sad"
                elif emotion == 5:
                    lb = "angry"
                elif emotion == 6:
                    lb = "fearful"
                elif emotion == 7:
                    lb = "disgust"
                elif emotion == 8:
                    lb = "surprised"
                else:
                    lb = "none"

                ravdess_db.loc[count] = [path, source, actor, gender, emotion, lb]
                count += 1
        ravdess_db.drop(ravdess_db.index[ravdess_db['emotion_lb'] == 'surprised'], inplace=True)
        ravdess_db.drop(ravdess_db.index[ravdess_db['emotion_lb'] == 'none'], inplace=True)
        ravdess_db.loc[ravdess_db.emotion_lb == 'calm', ['emotion', 'emotion_lb']] = 1, 'neutral'
        new_labels = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        ravdess_db = ravdess_db.replace({"emotion": new_labels})
        #ravdess_db.emotion_lb = ravdess_db.gender + "_" + ravdess_db.emotion_lb

        with open(RAVDESS_PATH, "wb") as file:
            pickle.dump(ravdess_db, file, protocol=pickle.HIGHEST_PROTOCOL)
    ravdess = pickle.load(open(RAVDESS_PATH, "rb"))
    return ravdess


def split_aibo_indices(last=25, split_ratio=0.7, seed=13):
    np.random.seed(seed)
    train_size = int(last * split_ratio)
    val_size = int((1 - split_ratio) / 2 * last)
    test_size = last - train_size - val_size

    a = np.arange(1, last + 1, 1)
    b = np.random.permutation(a)
    chunk_size = [train_size, val_size, test_size]
    np.cumsum(chunk_size)
    c = np.split(b, np.cumsum(chunk_size))
    train, val, test = c[0], c[1], c[2]
    if len(c[3]) > 0:
        print("some samples left unused while splitting")
    return train, val, test


def read_aibo():
    if not isfile(AIBO_PATH):
        print("AIBO not found. Creating AIBO dataset...")
        """
        df = pd.read_csv(os.path.join(DATA_DIR, aibo_labels_file), sep=" ", header=None,
                         names=["filename", "label", "percentage"])

        #labels_df = pd.DataFrame(data, columns=["source", "speaker", "n_rec", "num", "label", "percentage"], index=None)
        #labels_df['filename'] = labels_df.source + "_" + labels_df.speaker + "_" + labels_df.n_rec + "_" + labels_df.num
        #file_path = aibo_path + "wav/"
        #labels_df['path'] = file_path + labels_df['filename'] + ".wav"

        file_path = DATA_DIR + "/AIBO/wav/"
        df['path'] = file_path + df['filename'] + ".wav"
        df['session'] = df['filename'][0:3]

        train_indices, val_indices, test_indices = split_aibo_indices(32, 0.7, 6)
        df['split'] =df['speaker'].apply(
            lambda x: 'train' if x in train_indices else ('test' if x in test_indices else 'val'))
        """
        df = pd.read_csv(DATA_DIR + 'AIBO/aibo_labels_df.csv', header=0, index_col=0)
        with open(AIBO_PATH, "wb") as file:
            pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)
    aibo = pickle.load(open(AIBO_PATH, "rb"))
    return aibo


"""
def split_dataset_skip(dataset_features, dataset_labels, split_ratio=0.2):
    #Splitting dataset into train/val sets by taking every nth sample to val set
    skip_ratio = int(1/split_ratio)
    all_indexes = list(range(dataset_features.shape[0]))
    test_indexes = list(range(0, dataset_features.shape[0], skip_ratio))
    train_indexes = list(set(all_indexes) - set(test_indexes))
    val_indexes = train_indexes[::skip_ratio]
    train_indexes = list(set(train_indexes) - set(val_indexes))

    test_features = dataset_features[test_indexes]
    test_labels = dataset_labels[test_indexes]
    val_features = dataset_features[val_indexes]
    val_labels = dataset_labels[val_indexes]
    train_features = dataset_features[train_indexes]
    train_labels = dataset_labels[train_indexes]

    assert test_features.shape[0] == test_labels.shape[0]
    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return test_features, test_labels, val_features, val_labels, train_features, train_labels


def split_dataset_head(dataset_features, dataset_labels):
    #Splitting dataset into train/val sets by taking n first samples to val set
    val_features = dataset_features[:VAL_SIZE]
    val_labels = dataset_labels[:VAL_SIZE]
    train_features = dataset_features[VAL_SIZE:]
    train_labels = dataset_labels[VAL_SIZE:]

    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return val_features, val_labels, train_features, train_labels


def split_dataset_session_wise(dataset_features, dataset_labels, split_ratio=0.1):
    # Splitting dataset into train/val sets by taking every nth sample to val set
    #print("DATASET_FEATURES.SHAPE", dataset_features.shape)
    #print("DATASET_FEATURES.SHAPE[0]", dataset_features.shape[0])

    test_indexes = list(range(LAST_SESSION_SAMPLE_ID, dataset_features.shape[0]))
    print("test indexes:", test_indexes)

    skip_ratio = int(1/split_ratio)
    all_indexes = list(range(dataset_features.shape[0]))
    train_indexes = list(set(all_indexes) - set(test_indexes))

    val_indexes = train_indexes[::skip_ratio]
    print("val indexes:", val_indexes)
    print("len val indexes", len(val_indexes))
    train_indexes = list(set(train_indexes) - set(val_indexes))
    print("train indexes:", train_indexes)
    print("len train indexes", len(train_indexes))
    test_features = dataset_features[test_indexes]
    test_labels = dataset_labels[test_indexes]
    val_features = dataset_features[val_indexes]
    val_labels = dataset_labels[val_indexes]
    train_features = dataset_features[train_indexes]
    train_labels = dataset_labels[train_indexes]

    assert test_features.shape[0] == test_labels.shape[0]
    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]
    assert test_features.shape[0] + val_features.shape[0] + train_features.shape[0] == dataset_features.shape[0]
    return test_features, test_labels, val_features, val_labels, train_features, train_labels

def create_batches(test_features, test_labels, val_features, val_labels, train_features, train_labels, batch_size):
    test_iterator = BatchIterator(test_features, test_labels)
    train_iterator = BatchIterator(train_features, train_labels, batch_size)
    validation_iterator = BatchIterator(val_features, val_labels)
    return test_iterator, train_iterator, validation_iterator
"""


def load_or_create_dataset(create_func, features_path="", labels_path="", sessions_path="", **kwargs):
    # Extracting & Saving dataset
    if not isfile(features_path) or not isfile(labels_path) or not isfile(sessions_path):
        print("Dataset not found. Creating dataset...")
        create_func(**kwargs)
        print("Dataset created. Loading dataset...")

    # Loading dataset
    dataset = np.load(features_path)
    labels = np.load(labels_path)
    print("Dataset loaded.")
    assert dataset.shape[0] == labels.shape[0]
    sessions = np.load(sessions_path)
    return dataset, labels, sessions


@timeit
def create_spectrogram_dataset(**kwargs):
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    iemocap = read_iemocap()
    labels = []
    spectrograms = []
    sess_ids = []
    for i, sample in enumerate(iemocap):
        # for i, sample in iemocap.iterrows():
        labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]  # '1'
        sample_dir = "_".join(sample['File Name'].split("_")[:-1])  # 'Ses01F_impro01'
        sample_name = "{}.wav".format(sample['File Name'])  # 'Ses01F_impro01_F006.wav'
        # abs_path = join(IEMOCAP_FULL_PATH, "Session{}".format(session_id), "spectrograms/", sample_dir, sample_name)
        abs_path = join(DATA_DIR, "IEMOCAP/Session{}".format(session_id), sample_name)
        spectrogram = generate_spectrogram(abs_path, kwargs.get("view", False))
        spectrograms.append(spectrogram)
        sess_ids.append(session_id)
        if i % 100 == 0:
            print(i)
    np.save(SPECTROGRAMS_LABELS_PATH, np.array(labels))
    np.save(SPECTROGRAMS_FEATURES_PATH, np.array(spectrograms))
    np.save(SPECTROGRAMS_SESSIONS_PATH, np.array(sess_ids))


@timeit
def create_mspectr_dataset(**kwargs):
    iemocap = read_iemocap()
    mlabels = []
    mspectrograms = []
    sess_ids = []
    for i, sample in enumerate(iemocap):
        # labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]
        # sample_dir = "_".join(sample['File Name'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['File Name'])
        abs_path = join(DATA_DIR, "IEMOCAP/Session{}".format(session_id), sample_name)
        mspectrogram = generate_mel_spectr(abs_path, kwargs.get("view", False))
        if mspectrogram.shape[1] > ICP_MFCC_DUR:
            segments = [mspectrogram[:, y:y + ICP_MFCC_DUR] for y in range(0, mspectrogram.shape[1], MFCC_STEP)]
            for i in range(len(segments)):
                if segments[i].shape[1] is ICP_MFCC_DUR:  # skipping ends of mfcc shorter than 50
                    mspectrograms.append(segments[i])       # segments[i] shape  (128, 50)
                    mlabels.append(CLASS_TO_ID[sample['emotion']])
                    sess_ids.append(str(sample['File Name'][:5]))
        if i % 100 == 0:
            print("i:", i)
    np.save(MSPECTROGRAMS_SESSIONS_PATH, np.array(sess_ids))
    np.save(MSPECTROGRAMS_LABELS_PATH, np.array(mlabels))
    np.save(MSPECTROGRAMS_FEATURES_PATH, np.array(mspectrograms))


@timeit
def create_mfccs_dataset():
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    iemocap = read_iemocap()
    # labels = []
    # mfccs = []
    labels_segmented = []
    mfccs_segmented = []
    sess_ids = []

    for i, sample in enumerate(iemocap):
        # labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]
        # sample_dir = "_".join(sample['File Name'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['File Name'])
        abs_path = join(DATA_DIR, "IEMOCAP/Session{}".format(session_id), sample_name)
        mfcc = generate_mfcc(abs_path, view=False)

        # cut MFCCs into segments with 500 ms duration and 250 ms overlap
        if mfcc.shape[1] > ICP_MFCC_DUR:  # skipping mfccs less than 50 (are there no?)
            segments = [mfcc[:, y:y + ICP_MFCC_DUR] for y in range(0, mfcc.shape[1], MFCC_STEP)]
            for i in range(len(segments)):
                if segments[i].shape[1] is ICP_MFCC_DUR:  # skipping ends of mfcc shorter than 50
                    mfccs_segmented.append(segments[i])     # segments[i] shape  (13, 50) or (40, 50) if numcep = 40
                    #print("segments[i] shape ", segments[i].shape)
                    labels_segmented.append(CLASS_TO_ID[sample['emotion']])
                    sess_ids.append(session_id)
            # segment_labels = [CLASS_TO_ID[sample['emotion']] for i in range(len(segments))]
            # labels_segmented.append(np.transpose(segment_labels))
        # mfccs.append(mfcc)
        if i % 100 == 0:
            print("i:", i)
    np.save(MFCCS_LABELS_PATH, np.array(labels_segmented))
    np.save(MFCCS_SESSIONS_PATH, np.array(sess_ids))
    np.save(MFCCS_FEATURES_PATH, np.array(mfccs_segmented))
    """
    data, labels, sess = load_or_create_dataset(create_mspectr_dataset, MSPECTROGRAMS_FEATURES_PATH,
                                                MSPECTROGRAMS_LABELS_PATH, MSPECTROGRAMS_SESSIONS_PATH)
    num_ceps = 40 #Number of coefficients to extract
    cep_lifter = 42*2
    mfcc_tmp = [dct(10*np.log10(np.flipud(dt).T), type=2, axis=1, norm='ortho')[:num_ceps,:] for dt in data]
    mfcc = np.array([lifter(dt, cep_lifter) for dt in mfcc_tmp], dtype=float)
    
    np.save(MFCCS_LABELS_PATH, np.array(labels))
    np.save(MFCCS_SESSIONS_PATH, np.array(sess))
    np.save(MFCCS_FEATURES_PATH, np.array(mfcc))
    """


@timeit
def create_tbf_dataset():
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    iemocap = read_iemocap()
    # labels = []
    # mfccs = []
    labels = []
    tbfs = []
    sess_ids = []

    for i, sample in enumerate(iemocap):
        labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]
        # sample_dir = "_".join(sample['File Name'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['File Name'])
        abs_path = join(DATA_DIR, "IEMOCAP/Session{}".format(session_id), sample_name)
        tbf = generate_tbf(abs_path, view=False)
        tbfs.append(tbf)
        sess_ids.append(str(sample['File Name'][:5]))
        if i % 100 == 0:
            print("i:", i)
    np.save(TBF_LABELS_PATH, np.array(labels))
    np.save(TBF_SESSIONS_PATH, np.array(sess_ids))
    np.save(TBF_PATH, np.array(tbfs))


@timeit
def create_ravdess_dataset(input_type):
    if not isfile(os.path.join(RVD_AUDIOS_PATH)):
        print("RAVDESS Mel Spectrograms aren't extracted. Creating them...")
        modelling_db = read_ravdess()
        #audio_duration = 3
        #sampling_rate = 44100
        #input_length = sampling_rate * audio_duration
        #n_mfcc = 20
        #data_sample = np.zeros(input_length)
        #MFCC = librosa.feature.mfcc(data_sample, sr=sampling_rate, n_mfcc=n_mfcc)
        #audios = np.empty(shape=(modelling_db.shape[0], 128, MFCC.shape[1], 1))
        audios = np.empty(shape=(modelling_db.shape[0], 128, 259))
        # MFCC.shape[1] = 259
        #audios = np.empty([])
        count = 0
        #for i in tqdm(range(len(modelling_db))):
        for i in range(len(modelling_db)):
            if input_type=='mfcc':
                logspec = generate_ravdess_mfcc(modelling_db.path[i])
            else:
                logspec = generate_ravdess_mspectr(modelling_db.path[i])
            audios[count, ] = logspec
            count += 1
        np.save(RVD_AUDIOS_PATH, audios)
    return np.load(RVD_AUDIOS_PATH)


@timeit
def create_aibo_mfccs_dataset():
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    aibo = read_aibo()
    labels_segmented = []
    mfccs_segmented = []
    segments = []
    sess_ids = []

    for i, row in aibo.iterrows():
        #mfcc = generate_mfcc(aibo['path'][i], view=False).T
        mfcc = generate_mfcc(aibo['path'][i], view=False)
        # cut MFCCs into segments with 500 ms duration and 250 ms overlap
        if mfcc.shape[1] > ICP_MFCC_DUR:  # skipping mfccs less than 50 (are there no?)
            segments = [mfcc[:, y:y + ICP_MFCC_DUR] for y in range(0, mfcc.shape[1], MFCC_STEP)]
            for j in range(len(segments)):
                if segments[j].shape[1] is ICP_MFCC_DUR:  # skipping ends of mfcc shorter than 50
                    mfccs_segmented.append(segments[j])     # segments[i] shape  (13, 50) or (40, 50) if numcep = 40
                    #print("segments[i] shape ", segments[i].shape)
                    labels_segmented.append(row['lb'])
                    sess_ids.append(row['split'])
            # segment_labels = [CLASS_TO_ID[sample['emotion']] for i in range(len(segments))]
            # labels_segmented.append(np.transpose(segment_labels))
        # mfccs.append(mfcc)
        if i % 100 == 0:
            print("i:", i)
    #print(len(labels_segmented))
    #print(len(mfccs_segmented))
    np.save(AIBO_MFCCS_LABELS_PATH, np.array(labels_segmented))
    np.save(AIBO_MFCCS_SESSIONS_PATH, np.array(sess_ids))
    np.save(AIBO_MFCCS_FEATURES_PATH, np.array(mfccs_segmented))


def create_aibo_mspectr_dataset():
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    aibo = read_aibo()
    labels_segmented = []
    mspectr_segmented = []
    segments = []
    sess_ids = []

    for i, row in aibo.iterrows():
        #mfcc = generate_mel_spectr(aibo['path'][i], view=False).T
        mspectr = generate_mel_spectr(aibo['path'][i], view=False)
        # cut MFCCs into segments with 500 ms duration and 250 ms overlap
        if mspectr.shape[1] > ICP_MFCC_DUR:  # skipping mfccs less than 50 (are there no?)
            segments = [mspectr[:, y:y + ICP_MFCC_DUR] for y in range(0, mspectr.shape[1], MFCC_STEP)]
            for j in range(len(segments)):
                if segments[j].shape[1] is ICP_MFCC_DUR:  # skipping ends of mfcc shorter than 50
                    mspectr_segmented.append(segments[j])     # segments[i] shape  (13, 50) or (40, 50) if numcep = 40
                    #print("segments[i] shape ", segments[i].shape)
                    labels_segmented.append(row['lb'])
                    sess_ids.append(row['split'])
            # segment_labels = [CLASS_TO_ID[sample['emotion']] for i in range(len(segments))]
            # labels_segmented.append(np.transpose(segment_labels))
        # mfccs.append(mfcc)
        if i % 100 == 0:
            print("i:", i)
    #print(len(labels_segmented))
    #print(len(mspectr_segmented))
    np.save(AIBO_MSPECTR_LABELS_PATH, np.array(labels_segmented))
    np.save(AIBO_MSPECTR_SESSIONS_PATH, np.array(sess_ids))
    np.save(AIBO_MSPECTR_FEATURES_PATH, np.array(mspectr_segmented))


@timeit
def load_spectrogram_dataset():
    return load_or_create_dataset(create_spectrogram_dataset, SPECTROGRAMS_FEATURES_PATH, SPECTROGRAMS_LABELS_PATH,\
                                  SPECTROGRAMS_SESSIONS_PATH)


@timeit
def load_mspectr_dataset():
    return load_or_create_dataset(create_mspectr_dataset, MSPECTROGRAMS_FEATURES_PATH, MSPECTROGRAMS_LABELS_PATH, \
                                  MSPECTROGRAMS_SESSIONS_PATH)


@timeit
def load_mfcc_dataset():
    """
    data,ses=load_or_create_dataset(create_mspectr_dataset, MSPECTROGRAMS_FEATURES_PATH, MSPECTROGRAMS_LABELS_PATH)
    num_ceps=40 #Number of coefficients to extract
    cep_lifter=42*2
    mfcc_tmp = [dct(10*np.log10(np.flipud(dt).T), type=2, axis=1, norm='ortho')[:,:num_ceps] for dt in data ]
    mfcc = np.array([lifter(dt,cep_lifter) for dt in mfcc_tmp], dtype=float)
    """
    # test=lifter(mfcc_tmp[2000],cep_lifter)
    # plt.figure()
    # plt.title('Spectrogram')
    # plt.imshow(np.log(data[2000].T),aspect='auto')
    # plt.ylabel('Coefficients', fontsize=18)
    # plt.xlabel('Time [sec]', fontsize=18)
    # plt.tight_layout()
    # plt.show()
    # plt.figure()
    # plt.title('Spectrogram')
    # plt.imshow(mfcc_tmp[2000].T,aspect='auto')
    # plt.ylabel('Coefficients', fontsize=18)
    # plt.xlabel('Time [sec]', fontsize=18)
    # plt.tight_layout()
    # plt.show()
    #return mfcc, labels

    return load_or_create_dataset(create_mfccs_dataset, MFCCS_FEATURES_PATH, MFCCS_LABELS_PATH, MFCCS_SESSIONS_PATH)


@timeit
def load_tbf_dataset():
    return load_or_create_dataset(create_tbf_dataset, TBF_PATH, TBF_LABELS_PATH, TBF_SESSIONS_PATH)


@timeit
def load_ravdess_dataset():
    return load_or_create_dataset(create_ravdess_dataset(), RAVDESS_PATH)


@timeit
def load_aibo_mfccs_dataset():
    return load_or_create_dataset(create_aibo_mfccs_dataset, AIBO_MFCCS_FEATURES_PATH, AIBO_MFCCS_LABELS_PATH, AIBO_MFCCS_SESSIONS_PATH)


@timeit
def load_aibo_mspectr_dataset():
    return load_or_create_dataset(create_aibo_mspectr_dataset, AIBO_MSPECTR_FEATURES_PATH, AIBO_MSPECTR_LABELS_PATH, AIBO_MSPECTR_SESSIONS_PATH)


def balanced_sample_maker(X, y, random_seed=42):
    '''
    Take N of samples from each class
    :param X: X_train
    :param y: y_train
    :param sample_size: N of samples to take
    :param random_seed: save the randomness mode
    :return: resampled dataset
    '''
    uniq_levels = np.unique(y)
    #uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    # take sample size based on the class with the smallest amount
    sample_size = np.inf
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
        if len(obs_idx) < sample_size:
            sample_size = len(obs_idx)

    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        # replace=False ?
        balanced_copy_idx += over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train = X[balanced_copy_idx]
    labels_train = y[balanced_copy_idx]
    if len(data_train) == (sample_size * len(uniq_levels)):
        print('Number of samples: ', sample_size * len(uniq_levels), 'Number of samples per class: ', \
              sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('Number of samples is wrong!')

    labels, values = zip(*Counter(labels_train).items())
    #print('Number of classes: ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    if check:
        print('Good, all classes have the same number of examples')
    else:
        print('Repeat again your sampling, your classes are not balanced')
    # show the distribution between classes
    #indexes = np.arange(len(labels))
    #width = 0.5
    #plt.bar(indexes, values, width)
    #plt.xticks(indexes + width * 0.5, labels)
    #plt.show()
    return data_train, labels_train


def extract_norm_parameters(tensor):
    min, max = t.min(t.min(t.tensor(tensor), 2)[0], 0)[0], t.max(t.max(t.tensor(tensor), 2)[0], 0)[0]
    return min, max


def normalize(tensor, min, max):
    '''
    min/max are 1-D arrays with 128 values for mspectr
    :return: Normalized tensor pro 1 filter band
    '''
    #min, max = t.min(tensor), t.max(tensor)
    tensor = tensor.permute(0, 2, 1)
    tensor_norm = (tensor - min) / (max - min)
    tensor_norm = tensor_norm.permute(0, 2, 1)
    return tensor_norm


def split_ravdess(input_type='mfcc'):
    dataset_db = read_ravdess()
    dataset_db['split'] = np.where((dataset_db.actor == 23) | (dataset_db.actor == 24), 'Test',
                                   (np.where((dataset_db.actor == 19) | (dataset_db.actor == 22), 'Val', 'Train')))
    modelling_db = dataset_db[(dataset_db.split == 'Train') | (dataset_db.split == 'Val')]
    modelling_db.index = range(len(modelling_db.index))

    audios = create_ravdess_dataset(input_type)
    #print("audios", audios.shape)       # audios (1012, 128, 259)
    #print("dataset_db", dataset_db.shape)       # dataset_db (1012, 7)
    #print("modelling_db", modelling_db.shape)       # modelling_db (924, 7)

    x_train = audios[(dataset_db['split'] == 'Train')]
    y_train = dataset_db.emotion[(dataset_db['split'] == 'Train')]
    # y_train = modelling_db.emotion_lb[(modelling_db['split'] == 'Train')]
    #print(x_train.shape, y_train.shape)
    x_val = audios[(dataset_db['split'] == 'Val')]
    y_val = dataset_db.emotion[(dataset_db['split'] == 'Val')]

    x_test = audios[(dataset_db['split'] == 'Test')]
    y_test = dataset_db.emotion[(dataset_db['split'] == 'Test')]
    # y_test = modelling_db.emotion_lb[(modelling_db['split'] == 'Val')]
    #print(x_test.shape, y_test.shape)

    x_train = np.array(x_train)
    y_train = np.array(y_train, dtype=int)
    x_val = np.array(x_val)
    y_val = np.array(y_val, dtype=int)
    x_test = np.array(x_test)
    y_test = np.array(y_test, dtype=int)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data(dataset="iemocap", input_type="mfcc"):
    if dataset == "ravdess":
        x_train, y_train, x_val, y_val, x_test, y_test = split_ravdess(input_type)
        #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
        #print("y_test unique", np.unique(y_test))
        #print(type(x_train), type(y_train), type(x_val), type(y_val), type(x_test), type(y_test))
    elif dataset == "aibo":
        if input_type=="mfcc":
            my_x, my_y, split = load_aibo_mfccs_dataset()
            #my_x = np.load(AIBO_MFCCS_FEATURES_PATH)
            #my_y = np.load(AIBO_MFCCS_LABELS_PATH)
            #split = np.load(AIBO_MFCCS_SESSIONS_PATH)
        else:
            my_x, my_y, split = load_aibo_mspectr_dataset()
            #my_x = np.load(AIBO_MSPECTR_FEATURES_PATH)
            #my_y = np.load(AIBO_MSPECTR_LABELS_PATH)
            #split = np.load(AIBO_MSPECTR_SESSIONS_PATH)

        train_indexes, val_indexes, test_indexes = [],[],[]
        for i in range(len(split)):
            if split[i] == 'train':
                train_indexes.append(i)
            elif split[i] == 'val':
                val_indexes.append(i)
            else:
                test_indexes.append(i)
        x_train = my_x[train_indexes]
        y_train = my_y[train_indexes]
        x_val = my_x[val_indexes]
        y_val = my_y[val_indexes]
        x_test = my_x[test_indexes]
        y_test = my_y[test_indexes]
        print(x_train.shape, x_val.shape, x_test.shape)

    else:
        if input_type == "spectr":
            my_x, my_y = load_spectrogram_dataset()
            #split_index_val = 4290  # last sample from Session4
            #split_index_test = 6789
        elif input_type == "tbf":
            my_x, my_y = load_tbf_dataset()
            #split_index_val = 4290
            #split_index_test = 6789
        elif input_type == "mspectr":
            my_x, my_y, sess_ids = load_mspectr_dataset()
            #sess_ids = np.load(MSPECTROGRAMS_SESSIONS_PATH, allow_pickle=True)
            split_index_val = np.max(np.where(sess_ids == "Ses03"))     # take all Ses1-4 to training
            split_index_test = np.max(np.where(sess_ids == "Ses04"))
        else:
            my_x, my_y, sess_ids = load_mfcc_dataset()
            #print(my_x.shape)
            #sess_ids = np.load(MFCCS_SESSIONS_PATH, allow_pickle=True)
            split_index_val = np.max(np.where(sess_ids == '3'))
            split_index_test = np.max(np.where(sess_ids == '4'))
        #split_index_val = np.max(np.where(sess_ids == '3'))
        #split_index_test = np.max(np.where(sess_ids == '4'))
        all_indexes = list(range(my_x.shape[0]))
        # skip_ratio = int(1 / 0.2)  # split ratio = 0.2
        # test_indexes = list(range(0, my_x.shape[0], skip_ratio))
        # train_indexes = list(set(all_indexes) - set(test_indexes))
        # train_count = len(train_indexes)
        # test_count = len(test_indexes)

        test_indexes = all_indexes[(split_index_test + 1):]
        val_indexes = all_indexes[split_index_val:split_index_test]
        train_indexes = all_indexes[:(split_index_val + 1)]

        x_train = my_x[train_indexes]
        y_train = my_y[train_indexes]
        x_val = my_x[val_indexes]
        y_val = my_y[val_indexes]
        x_test = my_x[test_indexes]
        y_test = my_y[test_indexes]

    # sample same number of samples per class
    #x_train, y_train = balanced_sample_maker(x_train, y_train)
    #print("X-train shape", X_train.shape)      # mspectr: X-train shape (5000, 128, 50); mfcc: (5000, 40, 50)
    #x_val, y_val = balanced_sample_maker(x_val, y_val)
    #x_test, y_test = balanced_sample_maker(x_test, y_test)
    if dataset == "aibo":
        tensor_x_train_norm = t.Tensor(x_train)
        tensor_x_test_norm = t.Tensor(x_test)
        tensor_x_val_norm = t.Tensor(x_val)
    else:
        min, max = extract_norm_parameters(x_train)
        tensor_x_train_norm = normalize(t.Tensor(x_train), min, max).type(t.FloatTensor)
        tensor_x_test_norm = normalize(t.Tensor(x_test), min, max).type(t.FloatTensor)
        tensor_x_val_norm = normalize(t.Tensor(x_val), min, max).type(t.FloatTensor)

    #print(type(tensor_x_val_norm), type(t.LongTensor(y_val)))
    train_dataset = TensorDataset(tensor_x_train_norm, t.LongTensor(y_train))  # create your dataset
    # train_loader = DataLoader(my_dataset_train, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader
    valid_dataset = TensorDataset(tensor_x_val_norm, t.LongTensor(y_val))
    test_dataset = TensorDataset(tensor_x_test_norm, t.LongTensor(y_test))
    print(tensor_x_train_norm.shape, tensor_x_val_norm.shape, tensor_x_test_norm.shape)

    return train_dataset, valid_dataset, test_dataset, y_test
