import pandas as pd
from pandas import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score
matches = pd.read_csv("match_data.csv", parse_dates=['date'])

matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "",regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek
matches["is_home"] = (matches["venue"] == "Home").astype(int)
matches["target"] = matches["result"].map({"W": 1, "D": 0, "L": 0})
matches["result_points"] = matches["result"].map({"W": 3, "D": 1, "L": 0})
matches["league_weight"] = matches["comp"].map({"Premier League": 1.0, "Championship": 0.4}).fillna(0.5)
matches["year"] = matches["season"].str.split("-").str[1].astype(int)


def rolling_average(group, cols, new_cols, window=3):
    group = group.sort_values("date").copy()
    for col, new_col in zip(cols, new_cols):
        group[new_col] = group[col].shift(1).rolling(window=window, min_periods=1).mean()
    return group

cols = ["gf","ga","sh","sot","pk","pkatt"]
new_cols = [f"rolling_{col}" for col in cols]

frames = []
for team, group in matches.groupby("team", sort=False):
    frames.append(rolling_average(group, cols, new_cols, window=3))

matches_rolling = pd.concat(frames, ignore_index=True)

# Keep the prediction window to known seasons only until the 2026-27 data is complete.
matches_rolling = matches_rolling[matches_rolling["season"].isin(["2024-2025", "2025-2026"])].copy()

# Add stronger recent-form features based on the last 5 matches for each team.
matches_rolling = matches_rolling.sort_values(["team", "date"]).copy()

for col in ["result_points", "gf", "ga", "sh", "sot"]:
    matches_rolling[f"team_{col}_last_5"] = (
        matches_rolling.groupby("team")[col]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

matches_rolling["team_goal_diff_last_5"] = (
    matches_rolling["team_gf_last_5"] - matches_rolling["team_ga_last_5"]
)

# Opponent form from the previous performance of the opponent before this match date.
for col in ["result_points", "gf", "ga", "sh", "sot"]:
    matches_rolling[f"opp_{col}_last_5"] = (
        matches_rolling.groupby("opponent")[col]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

matches_rolling["opp_goal_diff_last_5"] = (
    matches_rolling["opp_gf_last_5"] - matches_rolling["opp_ga_last_5"]
)

# Normalise field names for the final model.
for old, new in {
    "opp_result_points_last_5": "opp_result_points",
    "opp_gf_last_5": "opp_gf",
    "opp_ga_last_5": "opp_ga",
    "opp_sh_last_5": "opp_sh",
    "opp_sot_last_5": "opp_sot",
}.items():
    if old in matches_rolling.columns:
        matches_rolling[new] = matches_rolling[old]

matches_rolling = matches_rolling.sort_values(["team", "date"]).reset_index(drop=True)

# Map prediction values to readable format
def pred_to_result(val):
    return {1: "W", 0: "D"}.get(val, "?")

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# Filter to only matches that have been played (have a result)
played_matches = matches_rolling[matches_rolling["target"].notna()].copy()
train_season = "2024-2025"
test_season = "2025-2026"

train = played_matches[played_matches["season"] == train_season].copy()
test_played = played_matches[played_matches["season"] == test_season].copy()

if train.empty or test_played.empty:
    raise ValueError(
        f"Training/test split requires {train_season} and {test_season} data. "
        f"Found {len(train)} train rows and {len(test_played)} test rows."
    )

predictors = ["venue_code", "opp_code", "hour", "day_code", "is_home"]
recent_form_predictors = [
    "team_result_points_last_5",
    "team_goal_diff_last_5",
    "team_gf_last_5",
    "team_ga_last_5",
    "team_sh_last_5",
    "team_sot_last_5",
    "opp_result_points",
    "opp_goal_diff_last_5",
    "opp_gf",
    "opp_ga",
    "opp_sh",
    "opp_sot",
]
new_predictors = predictors + recent_form_predictors + new_cols
train_weights = train["league_weight"].to_numpy()
rf.fit(train[new_predictors], train["target"], sample_weight=train_weights)

# Make predictions for all matches in the test season that have already been played
test_played = test_played.copy()
test_preds = rf.predict(test_played[new_predictors])
test_played["predicted_target"] = test_preds
test_played["predicted_result"] = test_played["predicted_target"].apply(pred_to_result)

acc = accuracy_score(test_played["target"], test_preds)
print(f"Accuracy on played matches in {test_season}: {acc:.2%}\n")

print("=" * 80)
print(f"PLAYED MATCHES ({test_season})")
print("=" * 80)
played_predictions = test_played[["date", "opponent", "result", "predicted_result", "year", "season"]]
print(played_predictions.to_string(index=False))

team_summary = pd.DataFrame({
    "team": test_played["team"].unique()
})

actual_counts = (
    test_played.groupby("team")["result"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={"W": "actual_wins", "D": "actual_draws", "L": "actual_losses"})
)
actual_counts = actual_counts.reindex(columns=["actual_wins", "actual_draws", "actual_losses"], fill_value=0)
actual_counts.index.name = "team"
actual_counts = actual_counts.reset_index()

predicted_counts = (
    test_played.groupby("team")["predicted_result"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={"W": "predicted_wins", "D": "predicted_draws", "L": "predicted_losses"})
)
predicted_counts = predicted_counts.reindex(columns=["predicted_wins", "predicted_draws", "predicted_losses"], fill_value=0)
predicted_counts.index.name = "team"
predicted_counts = predicted_counts.reset_index()

team_summary = actual_counts.merge(predicted_counts, on="team", how="outer")
team_summary["actual_games"] = (
    team_summary["actual_wins"] + team_summary["actual_draws"] + team_summary["actual_losses"]
)
team_summary["predicted_games"] = (
    team_summary["predicted_wins"] + team_summary["predicted_draws"] + team_summary["predicted_losses"]
)
team_summary["actual_win_pct"] = team_summary["actual_wins"] / team_summary["actual_games"]
team_summary["predicted_win_pct"] = team_summary["predicted_wins"] / team_summary["predicted_games"]
team_summary = team_summary[["team", "predicted_win_pct", "actual_win_pct"]].sort_values("team").reset_index(drop=True)
team_summary.to_csv(f"{test_season}_team_prediction_summary.csv", index=False)
print("\nSaved team summary to " + f"{test_season}_team_prediction_summary.csv")
print(team_summary.to_string(index=False))

# Predict on future matches only once the next season data is fully present.
future_matches = pd.DataFrame(columns=matches_rolling.columns)
if not future_matches.empty:
    rf.fit(train[predictors], train["target"])
    future_preds = rf.predict(future_matches[predictors])
    future_matches["predicted_target"] = future_preds
    future_matches["predicted_result"] = future_matches["predicted_target"].apply(pred_to_result)
    
    print("\n" + "=" * 80)
    print("UNPLAYED MATCHES (Future Predictions)")
    print("=" * 80)
    future_predictions = future_matches[["date","team", "opponent", "predicted_result", "year", "season"]]
    print(future_predictions.to_string(index=False))

acc = accuracy_score(test_played["target"], test_preds)
print(f"\nAccuracy on played matches in {test_season}: {acc:.2%}")
precision = precision_score(test_played["target"], test_preds, average='macro', zero_division=0)
print(f"Precision on played matches in {test_season}: {precision:.2%}")

# Keep output file generation for the known seasons only.
if 'future_predictions' in locals() and not future_predictions.empty:
    future_predictions.to_csv("prediction_data.csv", index=False)