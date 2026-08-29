from pathlib import Path
import re
from io import StringIO

import pandas as pd


HTML_DIR = Path(__file__).parent / "HTML"
OUTPUT_FILE = Path(__file__).parent / "match_data.csv"
ALLOWED_COMPETITIONS = {"Premier League", "Championship"}


def read_html_file(path):
    return path.read_text(encoding="utf-8")


def get_season(path, html):
    filename_match = re.match(r"(\d{4}-\d{4}) ", path.name)
    if filename_match:
        return filename_match.group(1)

    season_match = re.search(r"(?:for the |season )(\d{4}-\d{4})", html)
    if season_match:
        return season_match.group(1)

    raise ValueError(f"Could not determine season from {path.name}")


def get_team_name(stats_file):
    team_name = stats_file.name.split(" Stats,", 1)[0]
    return re.sub(r"^\d{4}-\d{4} ", "", team_name)


def get_competition(stats_file):
    match = re.search(r" Stats, (Premier League|Championship) \|", stats_file.name)
    return match.group(1) if match else None


def find_table(html, table_name):
    tables = pd.read_html(StringIO(html), match=table_name)
    if not tables:
        raise ValueError(f"Could not find the '{table_name}' table")

    table = tables[0]
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = table.columns.get_level_values(-1)
    return table


def find_shooting_file(stats_file, html_files):
    team_name = get_team_name(stats_file)
    season_match = re.match(r"^\d{4}-\d{4} ", stats_file.name)
    prefix = season_match.group(0) if season_match else ""
    expected_start = f"{prefix}{team_name} Match Logs (Shooting),"

    matches = [
        path for path in html_files
        if path.name.startswith(expected_start)
    ]
    return matches[0] if matches else None


def main():
    if not HTML_DIR.is_dir():
        raise FileNotFoundError(f"HTML directory not found: {HTML_DIR}")

    html_files = list(HTML_DIR.glob("*.html"))
    stats_files = sorted(
        path for path in html_files
        if " Stats," in path.name
        and not path.name.startswith("Premier League Stats")
    )

    if not stats_files:
        raise RuntimeError("No Premier League or Championship stats files found")

    all_matches = []

    for stats_file in stats_files:
        stats_html = read_html_file(stats_file)
        team_name = get_team_name(stats_file)
        season = get_season(stats_file, stats_html)
        shooting_file = find_shooting_file(stats_file, html_files)
        if shooting_file is None:
            print(
                f"Skipping {team_name} ({season}): "
                "matching shooting HTML file not found"
            )
            continue
        shooting_html = read_html_file(shooting_file)

        try:
            matches = find_table(stats_html, "Scores & Fixtures")
            shooting = find_table(shooting_html, "Shooting")
        except (ValueError, ImportError) as error:
            print(f"Skipping {stats_file.name}: {error}")
            continue

        required_shooting_columns = ["Date", "Sh", "SoT", "PK", "PKatt"]
        missing_columns = [
            column for column in required_shooting_columns
            if column not in shooting.columns
        ]
        if missing_columns:
            print(f"Skipping {stats_file.name}: missing {missing_columns}")
            continue

        # Some FBref exports omit Dist and FK. Keep the output schema stable.
        for column in ["Dist", "FK"]:
            if column not in shooting.columns:
                shooting[column] = pd.NA

        shooting_columns = [
            "Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"
        ]

        try:
            team_data = matches.merge(
                shooting[shooting_columns],
                how="left",
                on="Date",
            )
        except (KeyError, ValueError) as error:
            print(f"Skipping {stats_file.name}: merge failed: {error}")
            continue

        if "Comp" not in team_data.columns:
            print(f"Skipping {stats_file.name}: no competition column")
            continue

        team_data = team_data[
            team_data["Comp"].isin(ALLOWED_COMPETITIONS)
        ].copy()
        if team_data.empty:
            continue

        team_data["Season"] = season
        team_data["Team"] = team_name
        all_matches.append(team_data)

    if not all_matches:
        raise RuntimeError("No qualifying match data was extracted")

    match_df = pd.concat(all_matches, ignore_index=True)
    match_df.columns = [
        str(column).lower().replace(" ", "_")
        for column in match_df.columns
    ]
    
    # Convert date column to datetime
    if "date" in match_df.columns:
        match_df["date"] = pd.to_datetime(match_df["date"], format="%Y-%m-%d", errors="coerce")
    
    match_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {OUTPUT_FILE.name} with {len(match_df)} rows.")


if __name__ == "__main__":
    main()
