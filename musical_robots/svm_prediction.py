"""Implement SVM that predicts genre from an audio file."""

import pickle
from typing import Optional
import numpy as np
import pandas as pd
import librosa
import audioread


def svm_prediction(filename: str, genre_df: pd.DataFrame,
                   model_filename: str = "svm_model.pkl") -> Optional[str]:
    """
    Return SVM prediction of genre from an audio file.

    Args:
       filename (str): path to music mp3 file
       genre_df (pd.DataFrame): Dataframe containing genre information.
       model_filename (str): path to the model file. default is "svm_model.pkl"
    Returns:
        genre (str): predicted genre of the mp3 file
    """
    features = []

    try:
        y_audio, sample_rate = librosa.load(filename, sr=None, mono=True)
    except (RuntimeError, audioread.NoBackendError):
        print("Failed to load ", filename)
        return None
    except FileNotFoundError:
        print("File not found", filename)
        return None

    mel = librosa.feature.mfcc(y=y_audio, sr=sample_rate)
    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))

    spectral_contrast = librosa.feature.spectral_contrast(
        y_audio, sr=sample_rate, n_fft=2048, hop_length=1024
    )

    features.append((mel, spectral_contrast))

    test_data_array = np.array(
        [
            np.concatenate(
                [
                    np.average(features[0][0], axis=1),
                    np.median(features[0][0], axis=1),
                    np.std(features[0][0], axis=1),
                    [np.average(features[0][1])],
                    [np.median(features[0][1])],
                    [np.std(features[0][1])],
                ]
            )
        ]
    )

    with open(model_filename, "rb") as file:
        clf = pickle.load(file)

    label = clf.predict(test_data_array)

    genre = genre_df.iloc[label]["title"].item()

    return genre
