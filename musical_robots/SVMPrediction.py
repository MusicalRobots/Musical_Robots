import numpy as np
import pandas as pd
import librosa
import audioread
import pickle
from typing import Optional


def svm_prediction(filename: str, genre_df: pd.DataFrame) -> Optional[str]:
    """
    Return prediction
    Args:
       filename (str): path to music mp3 file
       genre_df: (pd.DataFrame) Dataframe containing genre information.
    Returns:
        genre (str): predicted genre of the mp3 file
    """

    features = []

    try:
        y, sr = librosa.load(filename, sr=None, mono=True)
    except (RuntimeError, audioread.NoBackendError):
        print('Failed to load ', filename)
        return None

    mel = librosa.feature.mfcc(y=y, sr=sr)
    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))

    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_fft=2048, hop_length=1024)

    features.append((mel, spectral_contrast))

    test_data_array = np.array([np.concatenate([np.average(features[0][0], axis=1), np.median(features[0][0], axis=1),
                                                np.std(features[0][0], axis=1), [np.average(features[0][1])],
                                                [np.median(features[0][1])], [np.std(features[0][1])]])])

    with open('svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    label = clf.predict(test_data_array)

    genre = genre_df.iloc[label]['title'].item()

    return genre
