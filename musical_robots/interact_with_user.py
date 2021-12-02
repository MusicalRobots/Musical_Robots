import numpy as np
import pandas as pd
import scipy
import ipywidgets as widgets
import IPython.display as ipd
import os
import librosa
import warnings
import audioread

from DatasetQueries import *
from SVMPrediction import svm_prediction
from SpectrogramDataset import SpectrogramDataset, MfccDataset, AudioFeature, create_audio_feature_dataset, create_mfcc_dataset, create_dataframes, create_dataset
from ipywidgets import interact, interact_manual
from IPython.display import Image, display, HTML, Audio, clear_output

class Interactive:
    def __init__(self):
        pass
    def uploader(self):
        uploader = widgets.FileUpload(multiple=False, description='Upload',
                              layout=widgets.Layout(width='250px', height='40px'))
        output=widgets.Output()
        display(uploader,output)
        return uploader, output



    def if_play_songs_first(self,y,sr,uploaded_filename):
        b1 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b2 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        b3 = widgets.Button(description='End', layout=widgets.Layout(width='30%'))
        display(HTML('<h2>Before analysing the genre, do you want to play your audio file?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b1, b2])
        display(h1, output)
        def yes_play_audio(b1):
            output.clear_output(wait=True)
            clear_output(wait=True)
            display(HTML('<h2>Gotcha! Gonna play the audio for you<h2>'.format(b1.description)))
            display(ipd.Audio(data=y, rate=sr))
            display(b3)
            
            b3.on_click(end_play_continue)


        def no_dont_play(b2):
            output.clear_output(wait=True) 
            clear_output(wait=True)
            display(HTML('<h2>Ok, analysing the genre for you <h2>'.format(b2.description)))
            
            return self.call_out_genre_prediction(output,uploaded_filename)


        def end_play_continue(b3):
            output.clear_output(wait=False)
            clear_output(wait=False)
            display(HTML('<h2>Starting analysing...<h2>'))
                        
            
            return self.call_out_genre_prediction(output,uploaded_filename)


        b1.on_click(yes_play_audio)
        b2.on_click(no_dont_play)
        

    
    def call_out_genre_prediction(self,output,uploaded_filename):
        output.clear_output(wait=False)
        b4 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b5 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        file_path_df, track_df, genre_df, total_genre_df = create_dataframes(file_paths_path = 'data/all_data_paths.txt' ,tracks_csv_path = 'data/fma_metadata/tracks.csv',genre_csv_path = 'data/fma_metadata/genres.csv')        
        genre = svm_prediction(uploaded_filename, genre_df)
        display(HTML('<h2>The genre is ...{}<h2>'.format(genre)))
        
        display(HTML('<h2> Do you want to know the most popular song in that genre and similiar genres?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b4, b5])
        display(h1, output)


        def yes_similiar_songs(b4):
            output.clear_output(wait=True)
            clear_output(wait=False)
            most_popular_song = return_most_popular_song(genre = genre, genre_df = total_genre_df, track_df=track_df)
            display(HTML('<h3>The most popular song in the genre {} is {} <h3>'.format(genre, most_popular_song)))
            similar_genres = return_similar_genres(genre = genre, genre_df = total_genre_df, track_df= track_df, k= 10) 
            display(HTML('<h3>The most similar genres to {} are : {} <h3><br>'.format(genre, similar_genres)))
            
            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))


        def no_similiar_songs(b5):
            output.clear_output(wait=True)
            clear_output(wait=False)

            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))


        b4.on_click(yes_similiar_songs)
        b5.on_click(no_similiar_songs)
        pass

def save_file(uploader):
        uploaded_filename = next(iter(uploader.value))
        content = uploader.value[uploaded_filename]['content']

    # save the uploaded file to the same directory , and load it in librosa
        with open(f'./{uploaded_filename}','wb') as fp:
            fp.write(content)
        warnings.simplefilter('ignore')
        with warnings.catch_warnings(record=False):
            try:
                y, sr = librosa.load(uploaded_filename)
                return y,sr,uploaded_filename

            except (RuntimeError, TypeError, audioread.NoBackendError):
                display(HTML('<h2>Sorry, the file you uploaded is not compatible, please check your music/audio file and upload again<h2>'))
        