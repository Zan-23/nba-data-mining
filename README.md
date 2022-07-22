#  NBA Play-by-Play Data mining analysis

In the scope of the Data mining praktikum at Technische Universität München (TUM), we analysed NBA play-by-play data,
as well as player information data which we manually acquired by scraping from the web.   

We performed an extensive analysis of both datasets and extracted meaningful insights and features from the data which we later used for game outcome prediction.
With our extracted features we managed to achieve a good accuracy score of around 65% on the test set with the 
same F1 score. The accuracy was higher when we only focused on the recent data meaning that we predicted game outcomes
mid season, there we managed to achieve a higher accuracy score of around 75%.

The team was made by us 4 students:
- Ege Arikan (egearikan)
- Gregor Caf (gregorcaf)
- Jonas Linder (Jonas-Lindner)
- Žan Stanonik (Zan-23)

# Table of Contents
1. [Getting started](#Getting started)
   1. [Environment setup](#Environment setup)
   2. [Data download](#Data download)
2. [Data sources](#Data sources )
3. [Presentation of project findings](#fourth-examplehttpwwwfourthexamplecom)

# Getting started
## Environment setup
To use the source code of the project we first have to install all the necessary 
dependencies. We achieve this with anaconda and the following commands, but 
you can also use pip:
1. Create a new dedicated conda environment:   
`conda create --name nba_dm python=3.9`
2. Activate the environment:   
`conda activate nba_dm`
3. Install all the required packages:   
`conda install --file requirements.txt`

Now we have everything installed we only need to acquire the data files needed for analysis.

## Data download
In the project we worked with two main data sets. The first one is the play-by-play data of which we 
acquired from [this website](https://eightthirtyfour.com/data), it contains all the play by play data from 
2008 to 2019. We have also written an script which automatically downloads the data from the website.

### Play-by-play data download 
Play by play data is downloaded by the script by default so you only need to call this script
from the project root directory:    
`python src/data_dowloading/data_downloader.py`

The script will output a lot of information during the scraping so that you can check 
the progress. At the end you should have 19 csv files in the `data/raw` directory.


### Player information data download
For this dataset we implemented data question from scratch, which means that the script extracts 
data directly from the [live NBA website](https://www.nba.com/stats/players/bio/).   
If you want to get the default data set (from 2000 to 2019) just run the script with this added flag:
`src/data_dowloading/data_downloader.py`

If you want to filter the data to a specific range of years you can use these two parameters:  
`src/data_dowloading/data_downloader.py 
--player_data --p_data_season_start=2002 --p_data_season_end=2016`

The given parameters mean that only the data from these seasons will be scraped. If you wish
to scrape data before 2000 or after 2019 you must change the source code of the allowed parameters,
since we didn't have time to properly test them on those seasons.   
The downloaded data will be present in the `data/raw/player_data` directory.


## Data sources 
Here we explicitly list the pages from which the data is scraped/downloaded:
- https://eightthirtyfour.com/data
- https://www.nba.com/stats/players/bio/

## Presentation of project findings
You can view a more detailed analysis of our project in the pdf presentation named 'final_presentation.pdf'.


