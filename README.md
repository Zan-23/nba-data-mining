#  NBA Play-by-Play Data mining
Repository for a our project which revolves around analyzing NBA play-by-play data for the Data Mininig Praktikum (at Technische Universität München)

## Environment setup
This is how you set up your environment with anaconda:
1. Create a new conda environment: `conda create --name nba_play_by_play_data_mining python=3.9`
2. Activate the environment: `conda activate nba_play_by_play_data_mining`
3. Install all the required packages: `conda install --file requirements.txt`

# Data download
The data with which we will work with is available on this [website](https://eightthirtyfour.com/data).
A script to download the data is already created and you can download the data 
with the following command: `python src/data/data_downloader.py`