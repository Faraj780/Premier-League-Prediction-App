import pandas as pd
from pandas import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
matches = pd.read_csv("match_data.csv", parse_dates=['date'])

matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "",regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = matches["result"].map({"W": 1, "D": 0, "L": -1})
matches["year"] = matches["season"].str.split("-").str[1].astype(int)

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# Filter to only matches that have been played (have a result)
played_matches = matches[matches["target"].notna()].copy()
train = played_matches[played_matches["date"] < "01-01-2026"]
test_played = played_matches[played_matches["date"] >= "01-01-2026"]

predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test_played[predictors])
acc = accuracy_score(test_played["target"], preds)
print(f"Accuracy on played matches: {acc:.2%}")
# Predict on future matches (matches without results)
future_matches = matches[matches["target"].isna()].copy()
if not future_matches.empty:
    future_preds = rf.predict(future_matches[predictors])
    future_matches["predicted_target"] = future_preds
    print("Future match predictions:")
    print(future_matches[["date", "team", "opponent", "predicted_target"]])