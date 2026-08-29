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

# Sort by date to ensure proper rolling calculations
matches = matches.sort_values("date").reset_index(drop=True)

# Add home/away advantage indicator
matches["is_home"] = (matches["venue"] == "Home").astype(int)

# ===== SHOOTING STATS FEATURES =====
# Fill NaN shooting stats with team average or 0
for col in ["sh", "sot", "dist", "fk", "pk", "pkatt"]:
    if col in matches.columns:
        matches[col] = matches[col].fillna(0)
        # Create opponent average (what opponent allows)
        matches[f"opp_{col}"] = matches.groupby("opponent")[col].shift().transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

# ===== FORM INDICATORS (Rolling Averages) =====
# Create rolling averages for each team grouped by team and sorted by date
for window in [5, 10, 20]:
    # Points from results: W=3, D=1, L=0
    matches["points"] = matches["target"].map({1: 3, 0: 1, -1: 0})
    
    # Rolling average points (form)
    matches[f"form_{window}"] = matches.groupby("team")["points"].shift().transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Rolling average goals for and against
    matches[f"gf_avg_{window}"] = matches.groupby("team")["gf"].shift().transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    matches[f"ga_avg_{window}"] = matches.groupby("team")["ga"].shift().transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Opponent's form (goals conceded by opponent's defense)
    matches[f"opp_form_{window}"] = matches.groupby("opponent")["points"].shift().transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    matches[f"opp_ga_avg_{window}"] = matches.groupby("opponent")["ga"].shift().transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

# ===== PROMOTED TEAMS ADJUSTMENT =====
# Identify teams promoted from Championship (teams that played in Championship in previous season)
promoted_teams = set()
for season in matches["season"].unique():
    current_year = int(season.split("-")[1])
    prev_season = f"{current_year-1}-{current_year}"
    
    # Check if previous season exists in data
    if prev_season not in matches["season"].values:
        continue
    
    # Teams in Championship last season
    championship_mask = (matches["season"] == prev_season) & (matches["comp"] == "Championship")
    championship_teams = set(matches[championship_mask]["team"].unique())
    
    # Teams in Premier League this season
    pl_mask = (matches["season"] == season) & (matches["comp"] == "Premier League")
    pl_teams = set(matches[pl_mask]["team"].unique())
    
    # Promoted teams are those in Championship last season but PL this season
    promoted_teams.update(pl_teams & championship_teams)

# Create promoted team indicator
matches["is_promoted"] = matches["team"].isin(promoted_teams).astype(int)
matches["opp_is_promoted"] = matches["opponent"].isin(promoted_teams).astype(int)

# Apply adjustment factor to form metrics for promoted teams (reduce by 15%)
for col in matches.columns:
    if "form_" in col or "avg_" in col:
        matches.loc[matches["is_promoted"] == 1, col] = (
            matches.loc[matches["is_promoted"] == 1, col] * 0.85
        )

# Fill NaN values with 0 for feature columns only (keep target NaNs for future matches)
feature_cols = [col for col in matches.columns if col not in ["target", "result"]]
for col in feature_cols:
    matches[col] = matches[col].fillna(0)

# Map prediction values to readable format
def pred_to_result(val):
    return {1: "W", 0: "D", -1: "L"}.get(val, "?")

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# Define expanded predictor set
predictors = [
    "venue_code", "opp_code", "hour", "day_code", "is_home",
    "sh", "sot", "dist", "fk", "pk", "pkatt",
    "opp_sh", "opp_sot", "opp_dist", "opp_fk", "opp_pk", "opp_pkatt",
    "form_5", "form_10", "form_20",
    "gf_avg_5", "gf_avg_10", "gf_avg_20",
    "ga_avg_5", "ga_avg_10", "ga_avg_20",
    "opp_form_5", "opp_form_10", "opp_form_20",
    "opp_ga_avg_5", "opp_ga_avg_10", "opp_ga_avg_20",
    "is_promoted", "opp_is_promoted"
]

# Filter to only matches that have been played (have a result)
played_matches = matches[matches["target"].notna()].copy()

# First: Train/test split for validation (to check model performance)
validation_train = played_matches[played_matches["date"] < "01-01-2026"]
validation_test = played_matches[played_matches["date"] >= "01-01-2026"]

rf_validation = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
rf_validation.fit(validation_train[predictors], validation_train["target"])

val_preds = rf_validation.predict(validation_test[predictors])
val_acc = accuracy_score(validation_test["target"], val_preds)
val_prec = precision_score(validation_test["target"], val_preds, average='weighted', zero_division=0)

print("=" * 80)
print("VALIDATION METRICS (Train on 2025, Test on 2026+)")
print("=" * 80)
print(f"Accuracy: {val_acc:.2%}")
print(f"Precision: {val_prec:.2%}\n")

# Second: Train on ALL played matches for best future predictions
rf.fit(played_matches[predictors], played_matches["target"])

# Make predictions using the full-data model on 2026+ matches
test_played = validation_test.copy()
test_preds = rf.predict(test_played[predictors])
test_played["predicted_target"] = test_preds
test_played["predicted_result"] = test_played["predicted_target"].apply(pred_to_result)

acc = accuracy_score(test_played["target"], test_preds)
prec = precision_score(test_played["target"], test_preds, average='weighted', zero_division=0)

print("=" * 80)
print("FULL-DATA MODEL METRICS (Trained on ALL played matches)")
print("=" * 80)
print(f"Accuracy on 2026+ matches: {acc:.2%}")
print(f"Precision on 2026+ matches: {prec:.2%}")
print(f"Training data: {len(played_matches)} total played matches\n")

print("Feature Importance (Top 15):")
feature_importance = pd.DataFrame({
    "feature": predictors,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(feature_importance.head(15).to_string(index=False))

print("\n" + "=" * 80)
print("PLAYED MATCHES (2025-2026 Season & 2026-2027 Season)")
print("=" * 80)
played_predictions = test_played[["date", "team", "opponent", "result", "predicted_result", "year"]]
print(played_predictions.to_string(index=False))

# Predict on future matches (matches without results)
future_matches = matches[matches["target"].isna()].copy()
if not future_matches.empty:
    future_preds = rf.predict(future_matches[predictors])
    future_matches["predicted_target"] = future_preds
    future_matches["predicted_result"] = future_matches["predicted_target"].apply(pred_to_result)
    
    print("\n" + "=" * 80)
    print("UNPLAYED MATCHES (Future Predictions)")
    print("=" * 80)
    future_predictions = future_matches[["date", "team", "opponent", "predicted_result", "year"]]
    print(future_predictions.to_string(index=False))
    
    # Save to CSV
    future_predictions.to_csv("prediction_data.csv", index=False)
    print(f"\nFuture predictions saved to 'prediction_data.csv' ({len(future_predictions)} matches)")
