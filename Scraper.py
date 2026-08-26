import requests
import time
import pandas
from bs4 import BeautifulSoup
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
data = requests.get(standings_url)
data.text
years = list (range(2026,2025,-1))
allmatches = []

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    try:
        standings_table = soup.select('table.stats_table')[0]
    except IndexError:
        print(f"No standings table found for year {year}. Closing.")
        break
    links = [l.get('href') for l in standings_table.find_all('a')]

    links = [l for l in links if '/squads/' in l]
    team_urls=["https://fbref.com/" + l for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = "https://fbref.com/" + previous_season

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", " ").replace("-")

        data = requests.get(team_url)
        matches = pandas.read_html(data.text, match="Scores & Fixtures")[0]

        soup = BeautifulSoup(data.text)
        links = [l.get('href') for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get("https://fbref.com" + links[0])
        shooting = pandas.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel(0)

        try:
            team_data = matches.merge(shooting[['Date','Sh','SoT','Dist','FK','PK','PKatt']], how="left", on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data['Comp'] == 'Premier League'or team_data['Comp'] == 'Championship']
        team_data['Season'] = year
        team_data['Team'] = team_name
        allmatches.append(team_data)

    time.sleep(1)

match_df = pandas.concat(allmatches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("match_data.csv")
