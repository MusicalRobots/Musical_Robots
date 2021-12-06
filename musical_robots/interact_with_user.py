"""Implementation of interactive class to communicate with user."""

import ipywidgets as widgets
import warnings
import ast

import pandas as pd

from DatasetQueries import *
from SVMPrediction import svm_prediction
from IPython.display import display, HTML, clear_output

from typing import Tuple


class Interactive:
    """Methods that create user interactive interfaces such as file uploader"""
    
    def __init__(self):
        pass

    def uploader(self) -> Tuple[widgets.FileUpload, widgets.Output]:
        """
        Create and display a file uploader.

        Returns:
            uploader (widgets.FileUpload):
            output (widgets.Output):
        """
        
        uploader = widgets.FileUpload(multiple=False, description='Upload',
                                      layout=widgets.Layout(width='250px', height='40px'))
        output = widgets.Output()
        display(uploader, output)
        
        return uploader, output

    def if_play_songs_first(self, y: np.ndarray, sr: int, uploaded_filename: str):
        """
        Create and display yes-or-no buttons for users to choose if they want to listen to the song first.

        Args:
            y (np.ndarray): Array containing the audio time series.
            sr (int): The sampling rate of y.
            uploaded_filename (str): The filename of the updated file.
        """
        
        b1 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b2 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        b3 = widgets.Button(description='End', layout=widgets.Layout(width='30%'))
        display(HTML('<h2>Before analysing the genre, do you want to play your audio file?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b1, b2])
        display(h1, output)

        def yes_play_audio(b1: widgets.Button):
            """If the users chooses yes, play the song."""
            
            output.clear_output(wait=True)
            clear_output(wait=True)
            display(HTML('<h2>Gotcha! Gonna play the audio for you! <h2>'.format(b1.description)))
            display(ipd.Audio(data=y, rate=sr))
            display(b3)
            
            b3.on_click(end_play_continue)

        def no_dont_play(b2: widgets.Button):
            """
            If the user chooses no, skip playing the song and start analysing.
           
            Returns:
                 self.call_out_genre_prediction: The call out genre prediction method
            """

            output.clear_output(wait=True) 
            clear_output(wait=True)
            display(HTML('<h2>Ok, analysing the genre for you <h2>'.format(b2.description)))
            
            return self.call_out_genre_prediction(output, uploaded_filename)

        def end_play_continue(b3: widgets.Button):
            """
            Create and display an end play and continue analyzing button.

            After the users chooses to play the song, create and display an end button for
            users to choose if they want to end playing the song and continue on analysing

            Returns:
                self.call_out_genre_prediction: The call out genre prediction method
            """
            
            output.clear_output(wait=False)
            clear_output(wait=False)
            display(HTML('<h2>Starting analysing...<h2>'))
            return self.call_out_genre_prediction(output, uploaded_filename)

        b1.on_click(yes_play_audio)
        b2.on_click(no_dont_play)

    def call_out_genre_prediction(self, output: widgets.Output, uploaded_filename: str) -> None:
        """
        Predict genre and ask users if they want to know the most popular song and similar genres.

        Args:
            output (widgets.Output):
            uploaded_filename (str): Filename of the uploaded file.
        """

        output.clear_output(wait=False)
        b4 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b5 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        
        file_path_df = pd.read_csv('./data/file_path_df', index_col=0)
        track_df = pd.read_csv('./data/track_df', index_col=0)
        genre_df = pd.read_csv('./data/genre_df', index_col=0)
        total_genre_df = pd.read_csv('./data/total_genre_df', index_col=0)

        def filter(row: pd.Series) -> List[int]:
            """
            Create filter that can be applied to each row of a pandas dataframe.

            Filter changes string representation of a List in column 'track_genres' to a List[int].

            Args:
                row (pd.Series): Row of a pandas dataframe.
            Return:
                (List[int]): Literal evaluation of string representation of List.
            """
            return [int(i) for i in ast.literal_eval(row['track_genres'])]

        track_df['track_genres'] = track_df.apply(filter, axis=1)

        genre = svm_prediction(uploaded_filename, genre_df)
        
        display(HTML('<h2>The genre is ...{}<h2>'.format(genre)))
        display(HTML('<h2> Do you want to know the most popular song in that genre and similar genres?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b4, b5])
        display(h1, output)

        def yes_similiar_songs(b4: widgets.Button) -> None:
            """If the users chooses yes, display the most popular songs and the most similar genres."""
            
            output.clear_output(wait=True)
            clear_output(wait=False)
            
            most_popular_song = return_most_popular_song(genre=genre, genre_df=total_genre_df, track_df=track_df)
            display(HTML('<h3>The most popular songs in the genre {} are {} <h3>'.format(genre, most_popular_song)))
            similar_genres = return_similar_genres(genre=genre, genre_df=total_genre_df, track_df=track_df, k=10)
            display(HTML('<h3>The most similar genres to {} are : {} <h3><br>'.format(genre, similar_genres)))
            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))

        def no_similiar_songs(b5: widgets.Button) -> None:
            """If the users choose no, interaction ends"""
            
            output.clear_output(wait=True)
            clear_output(wait=False)

            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))

        b4.on_click(yes_similiar_songs)
        b5.on_click(no_similiar_songs)
        pass


def save_file(uploader: widgets.FileUpload) -> Optional[Tuple[np.ndarray, int, str]]:
    """
    save a copy of the uploaded file and load it using Librosa.

    Args:
        uploader (widgets.FileUpload):

    Returns:
        y (np.ndarray): Array containing the audio time series
        sr (int): The sampling rate of y
        uploaded_filename (str): The filename of the uploaded file.
    """

    uploaded_filename = next(iter(uploader.value))
    content = uploader.value[uploaded_filename]['content']

    # save the uploaded file to the same directory , and load it in librosa
    with open(f'./{uploaded_filename}', 'wb') as fp:
        fp.write(content)
    warnings.simplefilter('ignore')
    with warnings.catch_warnings(record=False):
        try:
            y, sr = librosa.load(uploaded_filename)
            return y, sr, uploaded_filename

        except (RuntimeError, TypeError, audioread.NoBackendError):
            display(HTML('<h2>Sorry, the file you uploaded is not compatible, '
                         'please check your music/audio file and upload again<h2>'))
