import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup

def download_files_within_year_range(url, output_dir, start_year, end_year, file_extensions=None):
    """
    Downloads files from a webpage within a specific year range using wget.

    :param url: URL of the webpage to scrape.
    :param output_dir: Directory to save the downloaded files.
    :param start_year: Start year for filtering files.
    :param end_year: End year for filtering files.
    :param file_extensions: List of file extensions to filter (e.g., ['.nc']).
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the HTML content of the webpage
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links on the page
    links = [a['href'] for a in soup.find_all('a', href=True)]
    
    # Filter links by file extension if specified
    if file_extensions:
        pattern = re.compile(rf"({'|'.join(re.escape(ext) for ext in file_extensions)})$")
        links = [link for link in links if pattern.search(link)]
    
    # Filter links by year range
    year_pattern = re.compile(r'\b(19[8-9]\d|20[0-1]\d|2014)\b')  # Matches years 1980-2014
    filtered_links = [
        link for link in links if year_pattern.search(link)
        and start_year <= int(year_pattern.search(link).group()) <= end_year
    ]

    # Download each file using wget
    for link in filtered_links:
        # Handle relative URLs
        file_url = link if link.startswith('http') else requests.compat.urljoin(url, link)
        print(f"Downloading: {file_url}")
        try:
            subprocess.run(['wget', '-P', output_dir, file_url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {file_url}: {e}")

if __name__ == "__main__":
    # URL of the webpage containing file links
    webpage_url = "https://cirrus.ucsd.edu/~pierce/nonsplit_precip/precip/?C=N;O=A"

    # Directory to save the downloaded files
    output_directory = "/data/kas7897/Livneh/unsplit"

    # Year range for filtering files
    start_year = 1981
    end_year = 2014

    # File extensions to download (e.g., '.nc')
    extensions = ['.nc']

    # Run the downloader
    download_files_within_year_range(webpage_url, output_directory, start_year, end_year, extensions)
