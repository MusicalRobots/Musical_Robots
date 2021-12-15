# Musical Robot
[![Build Status](https://app.travis-ci.com/MusicalRobots/Musical_Robots.svg?branch=main)](https://app.travis-ci.com/MusicalRobots/Musical_Robots)
[![Coverage Status](https://coveralls.io/repos/github/MusicalRobots/Musical_Robots/badge.svg?branch=main)](https://coveralls.io/github/MusicalRobots/Musical_Robots?branch=main)

<p align="center">
	<img src="docs/MusicalRobotLogo2.png" width="320" height="300"> 
</p>

<p align="center">
A project that helps identify the genre of an mp3 music file and discover music of similar genres.
</p>

## Repository Structure
- /docs/: Component specification and functional specification.
- /musical_robots/data/: Raw data that was used for training the ML model and data containing track and genre information.
- /musical_robots/tests/: Unit tests.
- /musical_robots/demo.py: Musical Robot User Interaction
- /tutorial_notebooks/: Tutorial notebooks for SVM training and genre prediction.


## Use
1) Download full repository.
2) Download the training dataset 'fma_small.zip' and the datasets 'fma_metadata.zip" from https://github.com/mdeff/fma into the data folder. 
4) To use the service to identify an mp3 music file's genre and explore similar music:
	- Run the command "streamlit run musical_robots/demo.py" in terminal from the main repository.
	- The interaction will look as follows.
<p align="center">
	<img src="docs/MusicalRobotFlowchart.png" height="600"> 
</p>
	
3) To replicate the ML model:
	- Follow the tutorial in 'tutorial_notebooks/GenrePredictionTutorial.py'
4) To train your own ML model:
	- Follow the tutorial in "tutorial_notebooks/TrainSVMTutorial.ipynb".
5) To run the unit tests:
	- Run `python -m unittest discover -s musical_robots`  in the main directory.
	
## Acknowledgements
Thanks to Prof. David Beck and Anant Mittal at the University of Washington for teaching the course and guiding us during the execution of this project.
