import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn as skl
import sklearn.utils
import IPython.display as ipd
from sklearn.model_selection import train_test_split
import ast

from AudioDataset import SpectrogramDataset


file_paths = pd.read_csv('data/all_data_paths.txt', header = None, names = ['file_path'])
train_paths, test_paths = train_test_split(file_paths, test_size=0.33)

music_data = pd.read_csv('data/fma_metadata/tracks.csv', skiprows = [0,1,2],
                         usecols = [0, 6, 8, 11, 26, 39, 41, 44, 47, 52],
                         names = ['track_id', 'album_id', 'album_listens', 'album_title', 'artist_name',
                                 'track_favorites', 'track_genres', 'track_interest', 'track_listens', 'track_title'])

music_data['track_genres'] = music_data['track_genres'].apply(ast.literal_eval)

genre_df = pd.read_csv('data/fma_metadata/genres.csv', usecols=[0, 3])

train_data = SpectrogramDataset(train_paths, music_data, genre_df)
test_data = SpectrogramDataset(test_paths, music_data, genre_df)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

