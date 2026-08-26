import requests
import pandas
from bs4 import BeautifulSoup
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
data = requests.get(standings_url)
data.text
years = list (range(2026,2025),-1)
allmatches = []

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]
    links = [l.get('href') for l in standings_table.find_all('a')]
    