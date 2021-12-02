import numpy as np
import pandas as pd
import scipy
import ipywidgets as widgets
import IPython.display as ipd
import os
import librosa
import warnings
import audioread

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



    def if_play_songs_first(self,y,sr):
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
            display(HTML('<h2>Ok, gonna start analysing the genre for you <h2>'.format(b2.description)))
            return self.call_out_genre_prediction(output)


        def end_play_continue(b3):
            output.clear_output(wait=False)
            clear_output(wait=False)

            display(HTML('<h2>Starting analysing...<h2>'))
            return self.call_out_genre_prediction(output)


        b1.on_click(yes_play_audio)
        b2.on_click(no_dont_play)
        

    
    def call_out_genre_prediction(self,output):
        output.clear_output(wait=False)
        b4 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b5 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        display(HTML('<h2>The genre is ...<h2>'))
        display(HTML('<h2> Do you want to know the similiar songs?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b4, b5])
        display(h1, output)


        def yes_similiar_songs(b4):
            output.clear_output(wait=True)
            clear_output(wait=False)

            display(HTML('<h2>We have these similiar songs for you : .....<h2>'))
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
                return y,sr

            except (RuntimeError, TypeError, audioread.NoBackendError):
                display(HTML('<h2>Sorry, the file you uploaded is not compatible, please check your music/audio file and upload again<h2>'))