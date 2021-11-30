import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import ast

from SpectrogramDataset import SpectrogramDataset


class AE(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28)
        )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def train_AE(train_loader, learning_rate:float, epochs:int=100):
    model = AE(input_shape=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.view(-1, 784) #### CHANGE THIS NUM AS NECESSARY

            optimizer.zero_grad()
            outputs = model(inputs)

            train_loss = criterion(outputs, inputs)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        loss = loss/len(train_loader)

        print("epoch: {}/{}, loss = {:.6f}".format(epoch+1, epochs, loss))

    return None


def run_model():
    file_paths = pd.read_csv('data/all_data_paths.txt', header=None, names = ['file_path'])
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


    return None