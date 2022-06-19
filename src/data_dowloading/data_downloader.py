import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

NBA_PAGE_URL = "https://eightthirtyfour.com"
BASE_NBA_URL = "https://www.nba.com/stats/players/bio/"
NBA_DATA_PAGE_URL = f"{NBA_PAGE_URL}/data"
DATA_RAW_FOLDER = "./data/raw/"


def retrieve_data_urls_with_file_names():
    """
    Functions scrapes the https://eightthirtyfour.com/data/ page and returns an array of tuples with the url to the data
    and the file name.

    :return: Array of tuples. First element in the URL and the second is the file name.
    """
    print(f"Started retrieving urls from {NBA_DATA_PAGE_URL} html page ... \n")
    data_urls_arr = []
    # site doesn't have a valid SSL certificate, so we need to ignore the warning and put the flag off
    response = requests.get(NBA_DATA_PAGE_URL, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.findAll("a"):
        h_ref_value = link.get("href")

        # Check if the link ends with the right file extension
        if h_ref_value.endswith("pbp.csv") and "events" not in h_ref_value and "matthew" not in h_ref_value:
            current_url = NBA_PAGE_URL + h_ref_value.strip()
            data_urls_arr.append((current_url, h_ref_value.rsplit("/")[-1]))
        else:
            print(f"{link} is not a pbp.csv file")

    # Hardcoded number of files to download, it shouldn't change during the analysis
    if len(data_urls_arr) != 19:
        raise Exception(f"19 URLs were expected to be found, but the scraper found {len(data_urls_arr)}!")

    # Print the retrieved data
    print(f"\n Retrieved '{len(data_urls_arr)}' data links from {NBA_DATA_PAGE_URL}...")
    for counter, url_tuple in enumerate(data_urls_arr):
        current_url, file_name = url_tuple[0], url_tuple[1]
        print(f"{1 + counter:>3}. URL: {current_url}, FUTURE FILE NAME: '{file_name}'")

    return data_urls_arr


def download_and_save_url_data(data_urls_arr):
    """
    Method downloads the data from the given urls array and saves it to the raw folder. It also logs to the console
    to see the current progress.

    :param data_urls_arr: Array of tuples. First element in the URL and the second is the file name.
    """
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{current_time_str} - Started data download ... \n")

    for url, file_name in data_urls_arr:
        print(f"\nDownloading from {url} and saving to {DATA_RAW_FOLDER + file_name}")

        # Streaming, so we can iterate over the response.
        response = requests.get(url, stream=True, verify=False)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(DATA_RAW_FOLDER + file_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    execution_time = datetime.now() - current_time
    print(f"\nFinished data download in {execution_time}")


def get_player_info_season_urls():
    """
    Method generates all the urls that contain data about players for seasons 2000-2020.
    :return: Array of tuples. First element is the season (2002-2003) and the second values is the URL.
    """
    print(f"Started retrieving data  urls from '{BASE_NBA_URL}' website for seasons 2000-2020")

    url_arr = []
    # get all urls from 2000 to 2019
    for i in range(2000, 2019):
        season = f"{i}-{str(i+1)[2:]}"
        url = f"{BASE_NBA_URL}?Season={season}&SeasonType=Regular%20Season&PerMode=Totals"
        url_arr.append((season, url))
    return url_arr


def scrape_player_bios_and_save_to_csv():
    """
    Function scrapes the player bios from multiple urls, combines the information to dataframes, combines the dataframes
    for all seasons and save the information to a csv file.

    Function involves a lot of waiting, because the page needs to load up and it also contains a random sleep
    to avoid getting detected for scraping.
    """
    dir_path = f"{DATA_RAW_FOLDER}player-data"
    start_time = time.time()

    print(f"Started scraper for play info from '{BASE_NBA_URL}' website for season 2000-2020")
    if os.path.isdir(dir_path) is False:
        raise Exception(f"{dir_path} folder does not exist, but it should!")
    output_csv_file = f"{dir_path}/player_info.csv"
    url_arr = get_player_info_season_urls()

    main_df = pd.DataFrame()
    # Doesn't work with headless chrome, so we need to use a real browser, version 102 of chrome is needed
    s_count = 1
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    for season_str, season_url in url_arr:
        print(f"{s_count:02d}/{len(url_arr)} - Started scraping on '{season_url}' ...")
        driver.get(season_url)
        # wait to load the data
        driver.implicitly_wait(6)
        # Select to show all rows of players, gets the select element and changes the value to show all rows
        selector_page_el = driver.find_elements(by=By.XPATH,
                                                value="//select[contains(@class, 'stats-table-pagination__select')]")[0]
        players_selector = Select(selector_page_el)
        players_selector.select_by_visible_text("All")
        # wait for the data to load
        driver.implicitly_wait(5)

        # get the webpage source and find relevant elements
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        main_data_table = soup.find_all("div", {"class": "nba-stat-table"})[0]

        # get player IDs and name for each <a> element in table
        player_id_name_arr = []
        for a_el in soup.find_all("a", href=lambda value: value and value.startswith("/stats/player/")):
            # if there are duplicate slashes replace them with a single slash
            href_str = a_el["href"].replace("//", "/")
            if href_str.startswith("/stats/player/") and href_str.endswith("/"):
                player_id = href_str.split("/")[-2]
                if player_id.isdigit():
                    player_name = a_el.text.strip()
                    # because players can have the same name, so we can't use a dictionary
                    player_id_name_arr.append((player_id, player_name))
        # convert array to dataframe and prepare the columns for joining
        player_id_df = pd.DataFrame(player_id_name_arr)
        player_id_df.columns = ["Player ID", "Player"]

        # convert data from table to data frame and merge with player ID dataframe
        temp_df = pd.read_html(str(main_data_table))[0]
        temp_df["Player ID"] = np.nan
        for index, row in temp_df.iterrows():
            if temp_df.loc[index, "Player"] == player_id_df.loc[index, "Player"]:
                temp_df.loc[index, "Player ID"] = player_id_df.loc[index, "Player ID"]
            else:
                raise Exception(f"Player names on the same indexes should match!!")

        temp_df["Season"] = season_str
        main_df = pd.concat([main_df, temp_df], ignore_index=True)

        print(f"{s_count:02d}/{len(url_arr)} - Finished scraping on '{season_url}', performing random sleep ...")
        time.sleep(random.random() * 4)
        s_count += 1

    # quit the browser/selenium session
    driver.quit()
    print(f"\n\nSaving scraped data to a csv file {output_csv_file}")
    main_df.to_csv(output_csv_file)
    print(f"\nWhole scraping & saving process finished in {round(time.time() - start_time, 2)} seconds")


def scraper_main():
    # Code concerning the csv download
    # get urls from the data page
    data_urls_arr = retrieve_data_urls_with_file_names()
    # download and save the data
    download_and_save_url_data(data_urls_arr)

    # Code that scrapes the player bios
    scrape_player_bios_and_save_to_csv()
    print(f"End")


if __name__ == "__main__":
    scrape_player_bios_and_save_to_csv()

    # scraper_main()
