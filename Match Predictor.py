import pandas as pd
from pandas import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score
matches = pd.read_csv("match_data.csv", parse_dates=['date'])

matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "",regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = matches["result"].map({"W": 1, "D": 0, "L": -1})
matches["year"] = matches["season"].str.split("-").str[1].astype(int)



def rolling_average(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf","ga","sh","sot","pk","pkatt"]
new_cols = [f"rolling_{col}" for col in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_average(x, cols, new_cols),include_groups=False)

matches_rolling.index = range(matches_rolling.shape[0])


# Map prediction values to readable format
def pred_to_result(val):
    return {1: "W", 0: "D", -1: "L"}.get(val, "?")

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# Filter to only matches that have been played (have a result)
played_matches = matches_rolling[matches_rolling["target"].notna()].copy()
train = played_matches[played_matches["date"] < "01-01-2026"]
test_played = played_matches[played_matches["date"] >= "01-01-2026"]

predictors = ["venue_code", "opp_code", "hour", "day_code"]
new_predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
rf.fit(train[new_predictors], train["target"])

# Make predictions for all test matches (2026 onwards that have been played)
test_played = test_played.copy()
test_preds = rf.predict(test_played[new_predictors])
test_played["predicted_target"] = test_preds
test_played["predicted_result"] = test_played["predicted_target"].apply(pred_to_result)

acc = accuracy_score(test_played["target"], test_preds)
print(f"Accuracy on played matches from 2026 onwards: {acc:.2%}\n")

print("=" * 80)
print("PLAYED MATCHES (2025-2026 Season & 2026-2027 Season)")
print("=" * 80)
played_predictions = test_played[["date", "opponent", "result", "predicted_result", "year"]]
print(played_predictions.to_string(index=False))

# Predict on future matches (matches without results)
future_matches = matches[matches["target"].isna()].copy()
if not future_matches.empty:
    rf.fit(train[predictors], train["target"])
    future_preds = rf.predict(future_matches[predictors])
    future_matches["predicted_target"] = future_preds
    future_matches["predicted_result"] = future_matches["predicted_target"].apply(pred_to_result)
    
    print("\n" + "=" * 80)
    print("UNPLAYED MATCHES (Future Predictions)")
    print("=" * 80)
    future_predictions = future_matches[["date","team", "opponent", "predicted_result", "year"]]
    print(future_predictions.to_string(index=False))

acc = accuracy_score(test_played["target"], test_preds)
print(f"\nAccuracy on played matches from 2026 onwards: {acc:.2%}")
precision = precision_score(test_played["target"], test_preds, average='macro', zero_division=0)
print(f"Precision on played matches from 2026 onwards: {precision:.2%}")

future_predictions.to_csv("prediction_data.csv", index=False)