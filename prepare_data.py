import pandas as pd
import pickle
import numpy as np
import torch as t
from torch.utils.data import TensorDataset

from os.path import isfile, join

from preprocess_audio import generate_spectrogram, generate_mfcc, MFCC_DUR, MFCC_STEP
from utils import timeit


DATA_DIR = "/Users/el/embrace/data/"
#DATA_DIR = "/home/yq36elyb/data/"

BALANCED_IEMOCAP_PATH = DATA_DIR + 'iemocap_balanced.pkl'
SPECTROGRAMS_FEATURES_PATH = DATA_DIR + "spectrograms_features.npy"
SPECTROGRAMS_LABELS_PATH = DATA_DIR + "spectrograms_labels.npy"
MFCCS_FEATURES_PATH = DATA_DIR + "mfccs_features.npy"
MFCCS_LABELS_PATH = DATA_DIR + "mfccs_labels.npy"
MFCCS_SESSIONS_PATH = DATA_DIR + "mfccs_sessions.npy"

USED_CLASSES = ["neu", "hap", "sad", "ang", "fru", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "fru": 3, "ang": 4}
#USED_CLASSES = ["fru", "neu", "hap", "sad", "ang", "fea", "exc", "sur"]
#CLASS_TO_ID = {"neu": 0, "hap": 1, "sur": 2, "sad": 3, "fru": 4, "ang": 5, "fea": 6}
#ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
#ID_TO_FULL_CLASS = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Anger"}

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
    # Extracting IEMOCAP
    iemocap_path = DATA_DIR + 'iemocap.pkl'
    if not isfile(iemocap_path):
        print("IEMOCAP not found. Creating IEMOCAP dataset...")
        labels_df = pd.read_csv(DATA_DIR + "metadata.csv")
        # Create a column for gender
        labels_df['gender'] = labels_df.ID.str[-1:]
        # Convert gender to numeric
        labels_df['gender'] = labels_df['gender'].map(dict(zip(['M', 'F'], [0, 1])))
        #emotions = np.array(['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'fea', 'sur', 'dis'])
        #filt_df = labels_df[labels_df['emotion'].isin(emotions)]
        iemocap_list = labels_df[['File Name', 'emotion', 'gender']].copy()  # wav_file, label
        with open(DATA_DIR + 'iemocap.pkl', 'wb') as handle:
            pickle.dump(iemocap_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    iemocap = pickle.load(open(DATA_DIR + 'iemocap.pkl', "rb"))
    return iemocap


def create_balanced_iemocap():
    if not isfile(BALANCED_IEMOCAP_PATH):
        print("Balanced IEMOCAP not found. Creating balanced IEMOCAP dataset...")
        iemocap = read_iemocap()
        balanced_iemocap = []
        for index, row in iemocap.iterrows():
            # keep only 5 classes
            if row["emotion"] in USED_CLASSES:
                # merge excited in happy
                if row["emotion"] == "exc":
                    row["emotion"] = "hap"
                # oversample fear and surprise
                # data augmentation: add Gaussian noise to new samples
                #if row["emotion"] == "sur":
                 #   sur_df = row
                  #  for i in range(10):
                   #     balanced_iemocap.append(sur_df)
                #if row["emotion"] == "fea":
                 #   fear_df = row
                  #  for i in range(30):
                   #     balanced_iemocap.append(fear_df)
                balanced_iemocap.append(row)
        with open(BALANCED_IEMOCAP_PATH, "wb") as file:
            pickle.dump(balanced_iemocap, file, protocol=pickle.HIGHEST_PROTOCOL)
    balanced_iemocap = pickle.load(open(BALANCED_IEMOCAP_PATH, "rb"))
    return balanced_iemocap

""" After oversampling
fru    1849
neu    1708
hap    1636
fea    1240
sur    1177
ang    1103
sad    1084
"""
#Session1 for segments, not sample level: 40 fru, 20 hap, 10 fea: 10 fea, 10 fru, 10 hap

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
    #return split_dataset_session_wise(dataset, labels)
    return dataset, labels


@timeit
def create_spectrogram_dataset(**kwargs):
    #with open(balanced_iemocap_path, 'rb') as handle:
     #   iemocap = pickle.load(handle)
    iemocap = create_balanced_iemocap()
    labels = []
    spectrograms = []

    for i, sample in enumerate(iemocap):
    #for i, sample in iemocap.iterrows():
        labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]    # '1'
        sample_dir = "_".join(sample['File Name'].split("_")[:-1]) # 'Ses01F_impro01'
        sample_name = "{}.wav".format(sample['File Name']) # 'Ses01F_impro01_F006.wav'
        #abs_path = join(IEMOCAP_FULL_PATH, "Session{}".format(session_id), "spectrograms/", sample_dir, sample_name)
        abs_path = join(DATA_DIR, "Session{}".format(session_id), sample_name)
        spectrogram = generate_spectrogram(abs_path, kwargs.get("view", False))
        spectrograms.append(spectrogram)
        if (i % 100 == 0):
            print(i)

    np.save(SPECTROGRAMS_LABELS_PATH, np.array(labels))
    np.save(SPECTROGRAMS_FEATURES_PATH, np.array(spectrograms))


@timeit
def create_mfccs_dataset():
    #with open(balanced_iemocap_path, 'rb') as handle:
     #   iemocap = pickle.load(handle)
    iemocap = create_balanced_iemocap()
    #labels = []
    #mfccs = []
    labels_segmented = []
    mfccs_segmented = []
    sess_ids = []

    for i, sample in enumerate(iemocap):
        #labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['File Name'].split('Ses0')[1][0]
        #sample_dir = "_".join(sample['File Name'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['File Name'])
        abs_path = join(DATA_DIR, "Session{}".format(session_id), sample_name)
        mfcc = generate_mfcc(abs_path, view=False)

        # cut MFCCs into segments with 500 ms duration and 250 ms overlap
        if mfcc.shape[1] > MFCC_DUR:      # skipping mfccs less than 50 (are there no?)
            segments = [mfcc[:, y:y+MFCC_DUR] for y in range(0, mfcc.shape[1], MFCC_STEP)]
            for i in range(len(segments)):
                if segments[i].shape[1] is MFCC_DUR:        # skipping ends of mfcc shorter than 50
                    mfccs_segmented.append(segments[i])
                    labels_segmented.append(CLASS_TO_ID[sample['emotion']])
                    sess_ids.append(str(sample['File Name'][:5]))
            # segment_labels = [CLASS_TO_ID[sample['emotion']] for i in range(len(segments))]
            # labels_segmented.append(np.transpose(segment_labels))
        #mfccs.append(mfcc)
        if (i % 100 == 0):
            print("i:", i)

    np.save(MFCCS_LABELS_PATH, np.array(labels_segmented))
    np.save(MFCCS_SESSIONS_PATH, np.array(sess_ids))
    np.save(MFCCS_FEATURES_PATH, np.array(mfccs_segmented))


@timeit
def load_spectrogram_dataset():
    return load_or_create_dataset(create_spectrogram_dataset, SPECTROGRAMS_FEATURES_PATH, SPECTROGRAMS_LABELS_PATH)


@timeit
def load_mfcc_dataset():
    return load_or_create_dataset(create_mfccs_dataset, MFCCS_FEATURES_PATH, MFCCS_LABELS_PATH)


def normalize(tensor):
    min, max = t.min(tensor), t.max(tensor)
    tensor_norm = (tensor - min)/(max - min)
    return tensor_norm


def load_data(input="mfcc"):
    #transform = transforms.Compose([
     #   transforms.ToTensor(),
     #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #])

    #trainset = torchvision.datasets.CIFAR10(
     #   root=data_dir, train=True, download=True, transform=transform)
    #testset = torchvision.datasets.CIFAR10(
     #   root=data_dir, train=False, download=True, transform=transform)

    if input == "spectr":
        my_x, my_y = load_spectrogram_dataset()
        split_index = 4290  # last sample from Session4
    else:
        if not isfile(MFCCS_SESSIONS_PATH):
            print("Sessions indices not found. Generating them...")
            create_mfccs_dataset()
        my_x, my_y = load_mfcc_dataset()
        sess_ids = np.load(MFCCS_SESSIONS_PATH, allow_pickle=True)
        split_index = np.max(np.where(sess_ids == "Ses04"))

    all_indexes = list(range(my_x.shape[0]))
    # skip_ratio = int(1 / 0.2)  # split ratio = 0.2
    #test_indexes = list(range(0, my_x.shape[0], skip_ratio))
    #train_indexes = list(set(all_indexes) - set(test_indexes))
    #train_count = len(train_indexes)
    #test_count = len(test_indexes)

    test_indexes = all_indexes[(split_index+1):]
    train_indexes = all_indexes[:(split_index+1)]
    tensor_x_train = t.Tensor(my_x[train_indexes])  # transform to torch tensor
    tensor_x_train_norm = normalize(tensor_x_train)
    tensor_y_train = t.LongTensor(my_y[train_indexes])

    tensor_x_test = t.Tensor(my_x[test_indexes])  # transform to torch tensor
    tensor_x_test_norm = normalize(tensor_x_test)
    tensor_y_test = t.LongTensor(my_y[test_indexes])

    train_dataset = TensorDataset(tensor_x_train_norm, tensor_y_train)  # create your dataset
    #train_loader = DataLoader(my_dataset_train, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader
    test_dataset = TensorDataset(tensor_x_test_norm, tensor_y_test)

    return train_dataset, test_dataset
