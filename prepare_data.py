import pandas as pd
import pickle
import numpy as np
import torch as t
from torch.utils.data import TensorDataset

from os.path import isfile, join
from collections import Counter
import matplotlib.pyplot as plt

from preprocess_audio import generate_spectrogram, generate_mspectr, generate_mfcc, add_white_noise, MFCC_DUR, MFCC_STEP
from utils import timeit

SAMPLE_SIZE = 1000

DATA_DIR = "/Users/el/embrace/data/"
# DATA_DIR = "/home/yq36elyb/data/"

IEMOCAP_PATH = DATA_DIR + 'iemocap_with_noise.pkl'
SPECTROGRAMS_FEATURES_PATH = DATA_DIR + "spectrograms_features.npy"
SPECTROGRAMS_LABELS_PATH = DATA_DIR + "spectrograms_labels.npy"
MSPECTROGRAMS_FEATURES_PATH = DATA_DIR + "mspectrograms_features.npy"
MSPECTROGRAMS_LABELS_PATH = DATA_DIR + "mspectrograms_labels.npy"
MSPECTROGRAMS_SESSIONS_PATH = DATA_DIR + "mspectrograms_sessions.npy"
MFCCS_FEATURES_PATH = DATA_DIR + "mfccs_features.npy"
MFCCS_LABELS_PATH = DATA_DIR + "mfccs_labels.npy"
MFCCS_SESSIONS_PATH = DATA_DIR + "mfccs_sessions.npy"

USED_CLASSES = ["neu", "hap", "sad", "ang", "fru", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "fru": 3, "ang": 4}
# USED_CLASSES = ["fru", "neu", "hap", "sad", "ang", "fea", "exc", "sur"]
# CLASS_TO_ID = {"neu": 0, "hap": 1, "sur": 2, "sad": 3, "fru": 4, "ang": 5, "fea": 6}
# ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
# ID_TO_FULL_CLASS = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Anger"}

""" Full IEMOCAP value counts:
xxx    2507
fru    1849
neu    1708
ang    1103
sad    1084
exc    1041
hap     595
sur     107
fea      40
oth       3
dis       2
"""


def read_iemocap():
    # Extract IEMOCAP
    if not isfile(IEMOCAP_PATH):
        print("IEMOCAP not found. Creating IEMOCAP dataset...")
        labels_df = pd.read_csv(DATA_DIR + "metadata.csv")
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
            abs_path = join(DATA_DIR, "Session{}".format(session_id), file_name)
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

""" After oversampling
fru    1849
neu    1708
hap    1636
fea    1240
sur    1177
ang    1103
sad    1084
"""

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


def balanced_sample_maker(X, y, sample_size, random_seed=42):
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
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

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
    print('Number of classes: ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    #print(check)
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


def load_or_create_dataset(create_func, features_path, labels_path, **kwargs):
    # Extracting & Saving dataset
    if not isfile(features_path) or not isfile(labels_path):
        print("Dataset not found. Creating dataset...")
        create_func(**kwargs)
        print("Dataset created. Loading dataset...")

    # Loading dataset
    dataset = np.load(features_path)
    labels = np.load(labels_path)
    print("Dataset loaded.")
    assert dataset.shape[0] == labels.shape[0]
    # return split_dataset_session_wise(dataset, labels)
    return dataset, labels


@timeit
def create_spectrogram_dataset(**kwargs):
    # with open(balanced_iemocap_path, 'rb') as handle:
    #   iemocap = pickle.load(handle)
    iemocap = read_iemocap()
    labels = []
    spectrograms = []

    for i, sample in enumerate(iemocap):
        # for i, sample in iemocap.iterrows():
        labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]  # '1'
        sample_dir = "_".join(sample['File Name'].split("_")[:-1])  # 'Ses01F_impro01'
        sample_name = "{}.wav".format(sample['File Name'])  # 'Ses01F_impro01_F006.wav'
        # abs_path = join(IEMOCAP_FULL_PATH, "Session{}".format(session_id), "spectrograms/", sample_dir, sample_name)
        abs_path = join(DATA_DIR, "Session{}".format(session_id), sample_name)
        spectrogram = generate_spectrogram(abs_path, kwargs.get("view", False))
        spectrograms.append(spectrogram)
        if (i % 100 == 0):
            print(i)

    np.save(SPECTROGRAMS_LABELS_PATH, np.array(labels))
    np.save(SPECTROGRAMS_FEATURES_PATH, np.array(spectrograms))


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
        abs_path = join(DATA_DIR, "Session{}".format(session_id), sample_name)
        mspectrogram = generate_mspectr(abs_path, kwargs.get("view", False))
        if mspectrogram.shape[1] > MFCC_DUR:
            segments = [mspectrogram[:, y:y + MFCC_DUR] for y in range(0, mspectrogram.shape[1], MFCC_STEP)]
            for i in range(len(segments)):
                if segments[i].shape[1] is MFCC_DUR:  # skipping ends of mfcc shorter than 50
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
        abs_path = join(DATA_DIR, "Session{}".format(session_id), sample_name)
        mfcc = generate_mfcc(abs_path, view=False)

        # cut MFCCs into segments with 500 ms duration and 250 ms overlap
        if mfcc.shape[1] > MFCC_DUR:  # skipping mfccs less than 50 (are there no?)
            segments = [mfcc[:, y:y + MFCC_DUR] for y in range(0, mfcc.shape[1], MFCC_STEP)]
            for i in range(len(segments)):
                if segments[i].shape[1] is MFCC_DUR:  # skipping ends of mfcc shorter than 50
                    mfccs_segmented.append(segments[i])     # segments[i] shape  (13, 50) or (40, 50) if numcep = 40
                    #print("segments[i] shape ", segments[i].shape)
                    labels_segmented.append(CLASS_TO_ID[sample['emotion']])
                    sess_ids.append(str(sample['File Name'][:5]))
            # segment_labels = [CLASS_TO_ID[sample['emotion']] for i in range(len(segments))]
            # labels_segmented.append(np.transpose(segment_labels))
        # mfccs.append(mfcc)
        if i % 100 == 0:
            print("i:", i)

    np.save(MFCCS_LABELS_PATH, np.array(labels_segmented))
    np.save(MFCCS_SESSIONS_PATH, np.array(sess_ids))
    np.save(MFCCS_FEATURES_PATH, np.array(mfccs_segmented))


@timeit
def load_spectrogram_dataset():
    return load_or_create_dataset(create_spectrogram_dataset, SPECTROGRAMS_FEATURES_PATH, SPECTROGRAMS_LABELS_PATH)


@timeit
def load_mspectr_dataset():
    return load_or_create_dataset(create_mspectr_dataset, MSPECTROGRAMS_FEATURES_PATH, MSPECTROGRAMS_LABELS_PATH)


@timeit
def load_mfcc_dataset():
    return load_or_create_dataset(create_mfccs_dataset, MFCCS_FEATURES_PATH, MFCCS_LABELS_PATH)


def normalize(tensor):
    min, max = t.min(tensor), t.max(tensor)
    tensor_norm = (tensor - min) / (max - min)
    return tensor_norm


def load_data(input="mfcc"):
    # transform = transforms.Compose([
    #   transforms.ToTensor(),
    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # trainset = torchvision.datasets.CIFAR10(
    #   root=data_dir, train=True, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(
    #   root=data_dir, train=False, download=True, transform=transform)

    if input == "spectr":
        my_x, my_y = load_spectrogram_dataset()
        split_index = 4290  # last sample from Session4
    elif input == "mspectr":
        if not isfile(MFCCS_SESSIONS_PATH):
            print("Sessions indices not found. Generating them...")
            create_mfccs_dataset()
        my_x, my_y = load_mspectr_dataset()
        sess_ids = np.load(MSPECTROGRAMS_SESSIONS_PATH, allow_pickle=True)
        split_index = np.max(np.where(sess_ids == "Ses04"))     # take all Ses1-4 to training
    else:
        if not isfile(MFCCS_SESSIONS_PATH):
            print("Sessions indices not found. Generating them...")
            create_mfccs_dataset()
        my_x, my_y = load_mfcc_dataset()
        sess_ids = np.load(MFCCS_SESSIONS_PATH, allow_pickle=True)
        split_index = np.max(np.where(sess_ids == "Ses04"))

    all_indexes = list(range(my_x.shape[0]))
    # skip_ratio = int(1 / 0.2)  # split ratio = 0.2
    # test_indexes = list(range(0, my_x.shape[0], skip_ratio))
    # train_indexes = list(set(all_indexes) - set(test_indexes))
    # train_count = len(train_indexes)
    # test_count = len(test_indexes)

    test_indexes = all_indexes[(split_index + 1):]
    train_indexes = all_indexes[:(split_index + 1)]

    # sample same number of samples per class
    X_train, y_train = balanced_sample_maker(my_x[train_indexes], my_y[train_indexes], SAMPLE_SIZE)
    #print("X-train shape", X_train.shape)      # mspectr: X-train shape (5000, 128, 50); mfcc: (5000, 40, 50)
    X_test, y_test = balanced_sample_maker(my_x[test_indexes], my_y[test_indexes], SAMPLE_SIZE)

    tensor_x_train = t.Tensor(X_train)  # transform to torch tensor
    tensor_x_train_norm = normalize(tensor_x_train)
    tensor_y_train = t.LongTensor(y_train)

    tensor_x_test = t.Tensor(X_test)  # transform to torch tensor
    tensor_x_test_norm = normalize(tensor_x_test)
    tensor_y_test = t.LongTensor(y_test)

    train_dataset = TensorDataset(tensor_x_train_norm, tensor_y_train)  # create your dataset
    # train_loader = DataLoader(my_dataset_train, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader
    test_dataset = TensorDataset(tensor_x_test_norm, tensor_y_test)

    return train_dataset, test_dataset
