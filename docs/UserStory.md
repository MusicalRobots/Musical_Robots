# User Stories

## User 1

Ron has an audio file.  Ron wants to know the genre of the music file. Ron is in CSE 583 and knows what a Jupyter Notebook is.  Ron uploads his music file, and runs some commands in the notebook to return the genre.  Ron decided he wants to know the most popular songs in that genre.  He asks the system to return them.  Ron also wants to know what similar genres exist.  The system returns them.

**User:** Ron uploads a music file.  
**System:** Returns a figure displaying the probabilities that a clip belongs to genres.  The system also returns a message declaring the most likely genre. 
Display "Do you want to know the most popular songs in the genre?"
**Ron:** Says yes or no.
**System:** Display "Do you want to know the most similar genres?"
**Ron:** Says yes or no.
**System:** Display "Do you want to try another audio file"?
**Ron:** Says yes or no.


## User 2

Valentina wants to change the architecture of the network used to predict music genre and/or add data to the training set.  Valentina is a data scientist and highly technical.  

# Components

## Return most similar genres
Name: ReturnSimilarGenres
What it does: Returns up to top 10 most similar genres to the input genre.
Inputs: genre: string specifying genre to compare to
	number_similar: integer between 1 and 10 specifying how many ranked similar genres to return
Outputs: similar_genres: List of strings containing ranked list of most similar genres 

## Play song
Name: PlaySong
What it does: Plays a specified song. 
Inputs: mp3, wav, etc file?
	song name? 
Outputs: the librosa song listening thing 

## Predict genre
Name: PredictGenre
What it does: Predicts the genre of a given song.  If the input is a sound file, returns the prediction for that file.  If the input is a song song name, checks to see if the song is already in the database.  If not, tells you to upload a sound file.  
Inputs: mp3, wav, etc file?
	song name: string
Outputs: Plot displaying probabilities of genres. String specifying most likely genre. 



