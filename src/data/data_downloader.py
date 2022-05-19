import os
from datetime import datetime
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

NBA_PAGE_URL = "https://eightthirtyfour.com"
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


def scraper_main():
    os.path.join(os.path.dirname("./../.."))
    # get urls from the data page
    data_urls_arr = retrieve_data_urls_with_file_names()

    # download and save the data
    download_and_save_url_data(data_urls_arr)
    print(f"End")


if __name__ == "__main__":
    scraper_main()

