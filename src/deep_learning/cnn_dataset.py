"""
cnn_dataset.py

Dataset utilities for CNN-based Automatic Modulation Classification.
"""

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import joblib

TARGET_MODULATIONS = {

    "BPSK",

    "QPSK",

    "PAM4",

    "GFSK",

    "CPFSK",

    "AM-SSB",

    "AM-DSB",

    "QAM64",

    "8PSK"

}

MIN_SNR = 0

class RadioMLDataset(Dataset):

    def __init__(

        self,

        signals,

        labels

    ):

        self.signals = torch.tensor(

            signals,

            dtype=torch.float32

        )

        self.labels = torch.tensor(

            labels,

            dtype=torch.long

        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (

            self.signals[index],

            self.labels[index]

        )

def prepare_dataset(dataset):
    """
    Convert RadioML dictionary into
    CNN ready arrays.
    """

    X = []

    y = []

    snr_values = []

    for (modulation, snr), signals in dataset.items():

        if modulation not in TARGET_MODULATIONS:

            continue

        if snr < MIN_SNR:

            continue

        for signal in signals:

            X.append(signal)

            y.append(modulation)

            snr_values.append(snr)

    X = np.array(

        X,

        dtype=np.float32

    )

    y = np.array(y)

    snr_values = np.array(snr_values)

    return X, y, snr_values

def encode_labels(labels):

    encoder = LabelEncoder()

    encoded = encoder.fit_transform(labels)

    return encoded, encoder



def split_dataset(
    X,
    y,
    snr,
    train_size=0.8,
    random_state=42
):
    """
    Split into:
        Train      : 80%
        Validation : 10%
        Test       : 10%
    """

    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X,
        y,
        snr,
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp,
        y_temp,
        snr_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        snr_train,
        snr_val,
        snr_test
    )



def save_encoder(

    encoder,

    output_folder

):

    output_folder = Path(output_folder)

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        encoder,

        output_folder /

        "cnn_label_encoder.pkl"

    )